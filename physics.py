"""Controlled rigid-body benchmark. CPU simulation/rendering; no learned model.

Observations are RGB color centroids, not simulator state or segmentation IDs.
The two candidate geometry values and restitution values are known hypotheses.
Only the hidden hypothesis index is unknown to the predictor.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

SIZE = 128
SPAN = 0.8
DT = 1 / 960
QUERIES = ['Q1_speed', 'Q2_place_wall', 'Q3_remove_wall']


class World:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        self.calls = 0
        self.renders = 0

    def reset(self, y, e, wall_x=.2, start_x=-.25, speed=.25):
        self.calls += 1
        p.resetSimulation()
        p.setTimeStep(DT)
        p.setGravity(0, 0, 0)
        p.setPhysicsEngineParameter(numSolverIterations=100,
                                    restitutionVelocityThreshold=0,
                                    contactERP=0, contactSlop=0, deterministicOverlappingPairs=1)
        ball_c = p.createCollisionShape(p.GEOM_SPHERE, radius=.02)
        ball_v = p.createVisualShape(p.GEOM_SPHERE, radius=.02,
                                   rgbaColor=[1, 0, 0, 1])
        self.ball = p.createMultiBody(1, ball_c, ball_v, [start_x, 0, .04])
        wall_c = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.015, .03, .04])
        wall_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[.015, .03, .04],
                                   rgbaColor=[0, 0, 1, 1])
        self.wall = p.createMultiBody(0, wall_c, wall_v, [wall_x, y, .04])
        for obj, coeff in [(self.ball, e), (self.wall, 1.)]:
            p.changeDynamics(obj, -1, restitution=coeff, lateralFriction=0,
                             spinningFriction=0, rollingFriction=0,
                             linearDamping=0, angularDamping=0,
                             contactProcessingThreshold=0)
        p.resetBaseVelocity(self.ball, [speed, 0, 0])

    def image(self, top=False):
        self.renders += 1
        # Exact orthographic, untextured CPU rasterizer. Physics states are
        # available only to the renderer, not the observation consumer.
        # TinyRenderer custom orthographic was blank; distant perspective
        # leaked occasional subpixel depth cues and was rejected by validation.
        scale = 4
        im = Image.new('RGB', (SIZE*scale, SIZE*scale), 'white')
        draw = ImageDraw.Draw(im)
        def project(pos):
            vertical = pos[1] if top else pos[2]-.04
            return ((SIZE/2+pos[0]*SIZE/SPAN)*scale,
                    (SIZE/2-vertical*SIZE/SPAN)*scale)
        if self.wall >= 0:
            x,y = project(p.getBasePositionAndOrientation(self.wall)[0])
            hx = .015*SIZE/SPAN*scale
            hy = (.03 if top else .04)*SIZE/SPAN*scale
            draw.rectangle((x-hx,y-hy,x+hx,y+hy), fill='blue')
        x,y = project(p.getBasePositionAndOrientation(self.ball)[0])
        r = .02*SIZE/SPAN*scale
        draw.ellipse((x-r,y-r,x+r,y+r), fill='red')
        return np.asarray(im.resize((SIZE,SIZE), Image.Resampling.LANCZOS))

    def step(self):
        p.stepSimulation()
        return bool(p.getContactPoints(self.ball, self.wall)) if self.wall >= 0 else False

    def factual(self, y, e, wall_x, speed, save=False):
        self.reset(y, e, wall_x=wall_x, speed=speed)
        frames = []
        hit = False
        for i in range(round(.4 / DT)):
            hit |= self.step()
            if save and i % 96 == 95:
                frames.append(self.image())
        assert not hit, 'Factual clip must finish before any contact'
        return self.image(), self.image(top=True), frames

    def target(self, q, speed):
        if q == 0:
            p.resetBaseVelocity(self.ball, [speed, 0, 0])
        elif q == 1:
            pos, orn = p.getBasePositionAndOrientation(self.wall)
            p.resetBasePositionAndOrientation(self.wall, [pos[0], 0, pos[2]], orn)
        elif q == 2:
            p.removeBody(self.wall)
            self.wall = -1
        incoming = p.getBaseVelocity(self.ball)[0][0]
        hit = False
        trajectory = []
        for i in range(round(2 / DT)):
            hit |= self.step()
            if i % 48 == 47:
                trajectory.append(p.getBasePositionAndOrientation(self.ball)[0])
        return p.getBaseVelocity(self.ball)[0][0] / incoming, int(hit), trajectory

    def probe(self, e):
        # Separate reset and impact speed. Never execute the target intervention.
        self.reset(0, e, wall_x=.08, start_x=-.08, speed=.4)
        frames = []
        hit = False
        for i in range(round(.6 / DT)):
            hit |= self.step()
            if i + 1 in [round(.5 / DT), round(.6 / DT)]:
                frames.append(self.image())
        assert hit
        return frames, -p.getBaseVelocity(self.ball)[0][0] / .4


def centroid(rgb, color):
    idx = 0 if color == 'red' else 2
    others = [k for k in range(3) if k != idx]
    mask = (rgb[..., idx] > 180) & (rgb[..., others[0]] < 70) & (rgb[..., others[1]] < 70)
    yy, xx = np.where(mask)
    assert len(xx) >= 3, 'Object missing in RGB color detector'
    return np.array([xx.mean(), yy.mean()])


def generate(out, count=60, seed=20261103):
    out.mkdir(parents=True, exist_ok=True)
    examples = out / 'examples'
    examples.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    sim = World()
    specs, outputs, contacts, trajectories, means, validations = [], [], [], [], [], []
    started = time.perf_counter()
    for family in range(count):
        spec = dict(family=family, split=('dev' if family < 20 else 'validation' if family < 40 else 'test'),
                    wall_x=float(rng.uniform(.18, .22)), miss_y=float(rng.uniform(.075, .095)),
                    factual_speed=float(rng.uniform(.23, .27)), query_speed=float(rng.uniform(.5, .7)),
                    e_values=[float(rng.uniform(.15, .3)), float(rng.uniform(.65, .85))])
        specs.append(spec)
        vo, co, tr, obs, base_images, errors = [], [], [], [], [], []
        for g in range(2):  # g=0 miss; g=1 hit
            for ei, e in enumerate(spec['e_values']):
                y = spec['miss_y'] if g == 0 else 0
                world_v, world_c, world_t = [], [], []
                base, top, factual_frames = sim.factual(y, e, spec['wall_x'], spec['factual_speed'], True)
                base_images.append(np.stack(factual_frames+[base]))
                pf, recovered_e = sim.probe(e)
                errors.append(abs(recovered_e - e))
                obs.append([centroid(top, 'blue')[1],
                            centroid(pf[1], 'red')[0] - centroid(pf[0], 'red')[0],
                            centroid(base, 'blue')[1]])
                if family == 0:
                    tag = f'g{g}_e{ei}'
                    Image.fromarray(base).save(examples / f'{tag}_factual.png')
                    Image.fromarray(top).save(examples / f'{tag}_top.png')
                    for j, frame in enumerate(pf):
                        Image.fromarray(frame).save(examples / f'{tag}_probe{j}.png')
                    Image.fromarray(factual_frames[0]).save(examples / f'{tag}_factual.gif', save_all=True,
                            append_images=[Image.fromarray(f) for f in factual_frames[1:]], duration=100, loop=0)
                for q in range(3):
                    sim.factual(y, e, spec['wall_x'], spec['factual_speed'])
                    v, c, t = sim.target(q, spec['query_speed'])
                    world_v.append(v); world_c.append(c); world_t.append(t)
                    assert c == (int(g == 1) if q == 0 else int(q == 1))
                    expected = -e if c else 1.
                    errors.append(abs(v - expected))
                vo.append(world_v); co.append(world_c); tr.append(world_t)
        ambiguity = max(int(np.abs(im.astype(int) - base_images[0].astype(int)).max()) for im in base_images)
        assert ambiguity == 0, 'Base RGB leaks hidden-world identity'
        assert max(errors) < .01, f'Restitution mismatch: {max(errors)}'
        outputs.append(vo); contacts.append(co); trajectories.append(tr); means.append(obs)
        validations.append(dict(family=family, factual_max_rgb_difference=ambiguity,
                                max_restitution_error=max(errors)))
        if family % 10 == 0:
            print(f'Generated family {family}/{count}', flush=True)
    np.savez_compressed(out / 'worlds.npz', velocity=outputs, contact=contacts,
                        trajectory=trajectories, observation_mean=means)
    (out / 'families.json').write_text(json.dumps(specs, indent=2))
    (out / 'physics_validation.json').write_text(json.dumps(dict(
        seed=seed, families=count, dt=DT, image_size=SIZE, span=SPAN,
        renderer='Pillow supersampled orthographic RGB primitives', camera='exact orthographic',
        simulator='pybullet==3.2.7', host=platform.node(), simulations=sim.calls,
        renders=sim.renders, elapsed_seconds=time.perf_counter()-started,
        checks=validations), indent=2))
    p.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('data/pilot'))
    parser.add_argument('--families', type=int, default=60)
    args = parser.parse_args()
    generate(args.out, args.families)
