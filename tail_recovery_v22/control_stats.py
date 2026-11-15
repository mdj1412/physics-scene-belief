from stats22 import np,bootstrap,PAIRS,R,read,write,require,sha
require()
directory=R/"unstratified"
for f,h in read(directory/"freeze.json")["files"].items():assert sha(f)==h
delta=[];flips=[];all_loss=[]
for b in range(5):
    loss=np.load(R/f"evaluation_{b}.npz")["loss"].mean(-1)
    act=np.load(directory/f"choice_{b}.npz")["act"]
    ref=np.load(R/f"choices_{b}.npz")["acts"][:,:,:2]
    la=np.take_along_axis(loss,act[:,:,None]-1,axis=2)[:,:,0]
    lr=np.take_along_axis(loss,ref-1,axis=2)
    delta.append(lr-la[:,:,None]);flips.append((ref!=act[:,:,None]).mean(0));all_loss.append(la.mean(0))
ci,_=bootstrap(np.array(delta).reshape(5,512,-1),.05)
write(directory/"summary.json",dict(loss=np.mean(all_loss,0).tolist(),flips=np.mean(flips,0).tolist(),comparisons=ci))
