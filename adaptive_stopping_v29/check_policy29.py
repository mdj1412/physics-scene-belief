"""Independent numerical cross-check of lookahead weighting; run in v29 CPU environment."""
import numpy as np
from policy29 import (C_GRID,F_GRID,POST_GRID,RankFlipScorer,interpolate,
                      moments_for_horizon,outcome_coefficients,posterior_grid)


def main():
    rng=np.random.default_rng(29599061)
    factual=rng.normal(size=(20,2)); joint=rng.normal(size=(3,4)); ids=np.array([1,6,12])
    mass=posterior_grid(factual,ids,joint)
    # Evaluate individual points directly with all unprobed and paired rows.
    def logweight(i,j):
        ac,af=POST_GRID[i],POST_GRID[j]
        c=np.array([(1+ac)/2,(1-ac)/2]);f=np.array([(1+af)/2,(1-af)/2])
        unpaired=np.delete(factual,ids,axis=0)
        return (np.log(np.exp(unpaired)@c).sum()+
                sum(np.log(c@np.exp(row).reshape(2,2)@f) for row in joint))
    np.testing.assert_allclose(np.log(mass[2,7]/mass[68,78]),logweight(2,7)-logweight(68,78),atol=1e-12)
    score=rng.uniform(-.98,.98,(128,4))
    reference=np.linspace(-.9,.9,9)
    c0,f0=np.sum(POST_GRID*mass.sum(1)),np.sum(POST_GRID*mass.sum(0))
    tab=(np.sign(reference)*C_GRID[:,None,None]+F_GRID[None,:,None]-reference)
    np.testing.assert_allclose(interpolate(tab,c0,f0)[0],np.sign(reference)*c0+f0-reference)
    for horizon in (1,2,4):
        coef=outcome_coefficients(score[:,:horizon])
        mf,mcf=moments_for_horizon(mass,horizon)
        w=coef@mf[:-1]
        next_c=(coef@mcf)/w;next_f=(coef@mf[1:])/w
        # Brute-force posterior predictions with the actual product of likelihood ratios.
        for j in range(8):
            likelihood=np.prod(1+POST_GRID[:,None]*score[j,:horizon],axis=1)
            oracle=mass*likelihood
            np.testing.assert_allclose(w[j],oracle.sum(),atol=1e-12)
            np.testing.assert_allclose(next_c[j],(oracle.sum(1)@POST_GRID)/w[j],atol=1e-12)
            np.testing.assert_allclose(next_f[j],(oracle.sum(0)@POST_GRID)/w[j],atol=1e-12)
        s=RankFlipScorer.__new__(RankFlipScorer)
        s.table=tab; s.x_score=reference
        s.coefficients={horizon:coef}
        d=s.score(dict(alpha=[c0,f0]),horizon,factual,ids,joint)
        assert 0<d['flip_probability']<1 and d['priority']>0 and d['posterior_mean_grid_error']<1e-12
        s.table=np.ones_like(tab)
        d=s.score(dict(alpha=[c0,f0]),horizon,factual,ids,joint)
        assert d['priority']==0 and d['flip_probability']==0
    print('PASS: posterior factor replacement, multilinear risk lookup, joint predictive moments, nonflip and crossing scores')
if __name__=='__main__':main()
