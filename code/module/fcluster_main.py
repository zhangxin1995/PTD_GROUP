from .loadlocation import loadlocation
from .spatiotemporalmodel import generate_compare
import sys
import pickle as pkl
import os.path as osp
from collections import defaultdict
import numpy as np
import time
from scipy.spatial.distance import cdist,pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from module.eval_metrics import evaluate,rankscore,evaluate_args,eval_market1501_wang,eval_market1501_Huang,eval_market1501_Xie,eval_market1501_zhang,evaluate_group_search,tds2,average_search_time,eval_PTR_map, tc
import torch
from module.re_ranking import re_ranking
def norm(ff):
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff

def get_cluster_indices(cluster_assignments):                                           
    n = cluster_assignments.max()
    indices ={}
    for cluster_number in range(1, n + 1):
        indices[cluster_number-1]=np.where(cluster_assignments == cluster_number)[0]
    return indices


class GroupSearchBaseMethod:
    def __init__(self,gcs,gts,qcs,qts,g_world,compare,config):
        self.convert={'SQ0921':'A','SQ0922':'B','SQ0923':'C','SQ0924':'D','SQ0925':'E','SQ0926':'F','SQ0927':'G','SQ0928':'H','SQ0929':'I','SQ0930':'J','SQ0931':'K','SQ0932':'P'}
        self.gcs=gcs
        self.gts=gts
        self.qcs=qcs
        self.qts=qts
        self.g_world=g_world
        self.compare=compare
        self.config=config

    def forward(self,qfs,tfs,idx_from_gfs,threshold=0.039,topk=10,para=0):
        distmat=cdist(qfs,tfs,metric='cosine')
        sqs=[]
        sqs_world=[]
        for i in range(distmat.shape[0]):
            dist=distmat[i,:]
            args=np.argsort(dist)
            idxs=args[:topk]
            # idxs=np.where(dist<threshold)[0]
            sqs.append([(idx_from_gfs[k],dist[k]) for k in idxs])
        distmat2=np.zeros((len(sqs),len(sqs)))
        for i in range(len(sqs)):
            for j in range(len(sqs)):
                v=False
                distmat2[i,j]=self.dist_two_person(sqs[i],sqs[j],v=v,para=para)
            
        distmat2+=distmat2.transpose()
        distmat2/=4
        args=np.argsort(distmat2,axis=1)
        return distmat2,args

    def dist_two_person(self,sqs1,sqs2,v=False,para=0):
        max_d=100000
        if len(sqs1)==0 or len(sqs2)==0:
            return max_d
        # para=para
        dist=np.ones((len(sqs1),len(sqs2)))*10000
        max_d=self.config['max_d'] #10*7
        max_diff=self.config['max_diff'] #10*10
        u1=self.config['u1'] #0.1*12
        u2=self.config['u2'] #0.01*3
        eta=self.config['eta'] #1.5
        for i in range(len(sqs1)):
            for j in range(len(sqs2)):
                sqa=sqs1[i][0]
                s1=sqs1[i][1]
                sqb=sqs2[j][0]
                s2=sqs2[j][1]
                if sqa==sqb:
                    dist[i,j]=1e10
                else:
                    dist[i,j]=np.exp(eta*(s1+s2))*self.dist_two_trajectories(sqa,sqb,max_d,max_diff,u1,u2,v=v,para=para)
                # print(i,j,dist[i,j],s1,s2,self.dist_two_trajectories(sqa,sqb,max_d,max_diff,u1,u2,v=v,para=para))
        res=np.min(dist)
        return res

    def dtwk_shape(sekf,t,r):
        rows=t.shape[1]
        M = r.shape[0]
        N = t.shape[0]
        d = np.zeros((N,M))
        eps=1e-3
        if M==1:
            d=np.sum((t-np.tile(r,(N,1)))**2/max(M,N))
            return d,1
        elif N==1:
            d=np.sum((r-np.tile(t,(M,1)))**2/max(M,N))
            return d,1
        else:
            for i in range(rows):
                tt=t[:,i]
                rr=r[:,i]
                tt=(tt-np.mean(tt))/(np.std(tt,ddof=1)+eps)
                rr=(rr-np.mean(rr))/(np.std(rr,ddof=1)+eps)
                d+=(np.tile(tt.reshape(tt.shape[0],1),(1,M))-np.tile(rr.reshape(rr.shape[0],1).T,(N,1)))**2
        d=np.sqrt(d)
        if np.isnan(d).any():
            print(t)
            print(r)
            print(d)
            exit()
        D=np.zeros(d.shape)
        D[0,0]=d[0,0]
        for i in range(1,N):
            D[i,0]=D[i-1,0]+d[i,0]
        for i in range(1,M):
            D[0,i]=D[0,i-1]+d[0,i]
        for i in range(1,N):
            for j in range(1,M):
                D[i,j]=d[i,j]+min(D[i-1,j],min(D[i-1,j-1],D[i,j-1]))
        dist=D[N-1,M-1]
        i=N-1
        j=M-1
        path1=[]
        path2=[]
        k=1
        while i!=0 or j!=0:
            if i==0:
                j=j-1
            elif j==0:
                i-=1
            else:
                idx=np.argmin([D[i-1,j-1],D[i-1,j],D[i,j-1]])
                if idx==0:
                    i-=1
                    j-=1
                elif idx==1:
                    i-=1
                else:
                    j-=1
            path1.append(i)
            path2.append(j)
            k+=1
        return dist,k
    
    def dtwk(sekf,t,r):
        rows=t.shape[1]
        M = r.shape[0]
        N = t.shape[0]
        d = np.zeros((N,M))
        eps=1e-3
        for i in range(N):
            for j in range(M):
                d[i,j]=np.sum((t[i,:]-r[j,:])**2)
        d=np.sqrt(d)
        if np.isnan(d).any():
            print(t)
            print(r)
            print(d)
            exit()
        D=np.zeros(d.shape)
        D[0,0]=d[0,0]
        for i in range(1,N):
            D[i,0]=D[i-1,0]+d[i,0]
        for i in range(1,M):
            D[0,i]=D[0,i-1]+d[0,i]
        for i in range(1,N):
            for j in range(1,M):
                D[i,j]=d[i,j]+min(D[i-1,j],min(D[i-1,j-1],D[i,j-1]))
        dist=D[N-1,M-1]
        i=N-1
        j=M-1
        path1=[]
        path2=[]
        k=1
        while i!=0 or j!=0:
            if i==0:
                j=j-1
            elif j==0:
                i-=1
            else:
                idx=np.argmin([D[i-1,j-1],D[i-1,j],D[i,j-1]])
                if idx==0:
                    i-=1
                    j-=1
                elif idx==1:
                    i-=1
                else:
                    j-=1
            path1.append(i)
            path2.append(j)
            k+=1
        return dist,k
    def dist_two_trajectories(self,t1,t2,max_d,max_diff,u1,u2,v=False,para=0):
        seq1=[(self.gcs[j],self.gts[j]) for j in t1]
        seq2=[(self.gcs[j],self.gts[j]) for j in t2]
        seq1_world=[self.g_world[j][0] for j in t1]
        seq2_world=[self.g_world[j][0] for j in t2]
        dist=np.zeros((len(seq1),len(seq2)))
        c1=[self.gcs[j] for j in t1]
        c2=[self.gcs[j] for j in t2]
        flag=False
        for c in c1:
            if c in c2:
                flag=True
                break
        if not flag:
            return 1e10
        for u,p in enumerate(seq1):
            for w,q in enumerate(seq2):
                if p[0]==q[0]:
                    min_diff=abs(p[1]-q[1])/1000
                    if min_diff<max_diff:
                        d1=min_diff
                        pos1=np.array([seq1_world[u][key] for key in seq1_world[u].keys()])
                        # print(seq1_world[u].keys())
                        pos2=np.array([seq2_world[w][key] for key in seq2_world[w].keys()])
                        d2,k=self.dtwk(pos1,pos2)
                        dist[u,w]=u1*min_diff+u2*d2
                        # print(u1*min_diff,u2*d2)
                    else:
                        dist[u,w]=1e10
                else:
                    dist[u][w]=max_d
        N,M=dist.shape
        D=np.zeros(dist.shape)
        D[0,0]=dist[0,0]
        for i in range(1,N):
            D[i,0]=D[i-1,0]+dist[i,0]
        for i in range(1,M):
            D[0,i]=D[0,i-1]+dist[0,i]
        for i in range(1,N):
            for j in range(1,M):
                D[i,j]=dist[i,j]+min(D[i-1,j],min(D[i-1,j-1],D[i,j-1]))
        min_dist=D[N-1,M-1]
        # return min_dist
        i=N-1
        j=M-1
        path1=[]
        path2=[]
        k=1
        while i!=0 or j!=0:
            if i==0:
                j=j-1
            elif j==0:
                i-=1
            else:
                idx=np.argmin([D[i-1,j-1],D[i-1,j],D[i,j-1]])
                if idx==0:
                    i-=1
                    j-=1
                elif idx==1:
                    i-=1
                else:
                    j-=1
            path1.append(i)
            path2.append(j)
            k+=1
        if k==0:
            return 1e10
        return min_dist/k

    def trajectory_cluster(self,idx_from_gfs,gcs,gts,threshold=10,alpha=10,beta=1):
        convert={'SQ0921':'A','SQ0922':'B','SQ0923':'C','SQ0924':'D','SQ0925':'E','SQ0926':'F','SQ0927':'G','SQ0928':'H','SQ0929':'I','SQ0930':'J','SQ0931':'K','SQ0932':'P'}
        idxs=range(len(idx_from_gfs))
        sqs=[]
        for i in idxs:
            seq=[(gcs[j],gts[j]) for j in idx_from_gfs[i]]
            sqs.append(seq)
        distmat=np.ones((len(idxs),len(idxs)))
        for i in range(len(idxs)):
            for j in range(i+1,len(idxs)):
                seq1=''
                for p in sqs[i]:
                    seq1+=convert[p[0]]
                seq2=''
                for p in sqs[j]:
                    seq2+=convert[p[0]]
                r=1-lv.ratio(seq1,seq2)
                dist=np.zeros((len(seq1),len(seq2)))
                for u,p in enumerate(sqs[i]):
                    for w,q in enumerate(sqs[j]):
                        if p[0]==q[0]:
                            min_diff=abs(p[1]-q[1])/1000.0
                            if min_diff<60:
                                dist[u][w]=min_diff
                            else:
                                dist[u][w]=max_d
                        else:
                            dist[u][w]=max_d
                ind=linear_assignment(dist)
                d=[]
                for pair in ind:
                    if dist[pair[0]][pair[1]]<max_d:
                        #print(pair[0],pair[1],dist[pair[0]][pair[1]])
                        d.append(dist[pair[0]][pair[1]])
                if len(d)==0 :
                    d=max_d
                else:
                    d=sum(d)/len(d)
                distmat[i,j]=alpha*r+beta*d
        distmat+=distmat.transpose()
        link = linkage(distmat, "average")
        clusters = fcluster(link,threshold, criterion='distance')
        new_idxs=defaultdict(list)
        group_keys={}
        for i,c in enumerate(clusters):
            new_idxs[c-1].append(i)
            group_keys[i]=c-1
        return group_keys,new_idxs
    def __init__(self,gcs,gts,qcs,qts,g_world,compare):
        self.convert={'SQ0921':'A','SQ0922':'B','SQ0923':'C','SQ0924':'D','SQ0925':'E','SQ0926':'F','SQ0927':'G','SQ0928':'H','SQ0929':'I','SQ0930':'J','SQ0931':'K','SQ0932':'P'}
        self.gcs=gcs
        self.gts=gts
        self.qcs=qcs
        self.qts=qts
        self.g_world=g_world
        self.compare=compare

    def forward(self,qfs,tfs,idx_from_gfs,config,topk=1,para=0):
        threshold=config['threshold1']
        distmat=cdist(qfs,tfs,metric='cosine')
        sqs=[]
        sqs_world=[]
        for i in range(distmat.shape[0]):
            dist=distmat[i,:]
            idxs=np.where(dist<threshold)[0]
            sqs.append([(idx_from_gfs[k],dist[k]) for k in idxs])
        distmat2=np.zeros((len(sqs),len(sqs)))
        for i in range(len(sqs)):
            for j in range(len(sqs)):
                v=False
                distmat2[i,j]=self.dist_two_person(sqs[i],sqs[j],config,v=v,para=para)
        distmat2+=distmat2.transpose()
        distmat2/=4
        # print(distmat2)
        args=np.argsort(distmat2,axis=1)
        return distmat2,args

    def dist_two_person(self,sqs1,sqs2,config,v=False,para=0):
        max_d=100000
        if len(sqs1)==0 or len(sqs2)==0:
            return max_d
        max_d=10*5
        # para=para
        dist=np.zeros((len(sqs1),len(sqs2)))
        max_diff=1000
        u1=config['u1'] #0.1*16
        u2=config['u2'] #0.01*2
        eta=config['eta'] #0.1*7
        for i in range(len(sqs1)):
            for j in range(len(sqs2)):
                sqa=sqs1[i][0]
                s1=sqs1[i][1]
                sqb=sqs2[j][0]
                s2=sqs2[j][1]
                if sqa==sqb:
                    dist[i,j]=max_d
                else:
                    dist[i,j]=np.exp(eta*(s1+s2))*self.dist_two_trajectories(sqa,sqb,max_d,max_diff,u1,u2,v=v,para=para)
        return np.min(dist)

    def dtwk_shape(sekf,t,r):
        rows=t.shape[1]
        M = r.shape[0]
        N = t.shape[0]
        d = np.zeros((N,M))
        eps=1e-3
        if M==1:
            d=np.sum((t-np.tile(r,(N,1)))**2/max(M,N))
            return d,1
        elif N==1:
            d=np.sum((r-np.tile(t,(M,1)))**2/max(M,N))
            return d,1
        else:
            for i in range(rows):
                tt=t[:,i]
                rr=r[:,i]
                tt=(tt-np.mean(tt))/(np.std(tt,ddof=1)+eps)
                rr=(rr-np.mean(rr))/(np.std(rr,ddof=1)+eps)
                d+=(np.tile(tt.reshape(tt.shape[0],1),(1,M))-np.tile(rr.reshape(rr.shape[0],1).T,(N,1)))**2
        d=np.sqrt(d)
        if np.isnan(d).any():
            print(t)
            print(r)
            print(d)
            exit()
        D=np.zeros(d.shape)
        D[0,0]=d[0,0]
        for i in range(1,N):
            D[i,0]=D[i-1,0]+d[i,0]
        for i in range(1,M):
            D[0,i]=D[0,i-1]+d[0,i]
        for i in range(1,N):
            for j in range(1,M):
                D[i,j]=d[i,j]+min(D[i-1,j],min(D[i-1,j-1],D[i,j-1]))
        dist=D[N-1,M-1]
        i=N-1
        j=M-1
        path1=[]
        path2=[]
        k=1
        while i!=0 or j!=0:
            if i==0:
                j=j-1
            elif j==0:
                i-=1
            else:
                idx=np.argmin([D[i-1,j-1],D[i-1,j],D[i,j-1]])
                if idx==0:
                    i-=1
                    j-=1
                elif idx==1:
                    i-=1
                else:
                    j-=1
            path1.append(i)
            path2.append(j)
            k+=1
        return dist,k
    
    def dtwk(sekf,t,r):
        rows=t.shape[1]
        M = r.shape[0]
        N = t.shape[0]
        d = np.zeros((N,M))
        eps=1e-3
        for i in range(N):
            for j in range(M):
                d[i,j]=np.sum((t[i,:]-r[j,:])**2)
        d=np.sqrt(d)
        if np.isnan(d).any():
            print(t)
            print(r)
            print(d)
            exit()
        D=np.zeros(d.shape)
        D[0,0]=d[0,0]
        for i in range(1,N):
            D[i,0]=D[i-1,0]+d[i,0]
        for i in range(1,M):
            D[0,i]=D[0,i-1]+d[0,i]
        for i in range(1,N):
            for j in range(1,M):
                D[i,j]=d[i,j]+min(D[i-1,j],min(D[i-1,j-1],D[i,j-1]))
        dist=D[N-1,M-1]
        i=N-1
        j=M-1
        path1=[]
        path2=[]
        k=1
        while i!=0 or j!=0:
            if i==0:
                j=j-1
            elif j==0:
                i-=1
            else:
                idx=np.argmin([D[i-1,j-1],D[i-1,j],D[i,j-1]])
                if idx==0:
                    i-=1
                    j-=1
                elif idx==1:
                    i-=1
                else:
                    j-=1
            path1.append(i)
            path2.append(j)
            k+=1
        return dist,k
    def dist_two_trajectories(self,t1,t2,max_d,max_diff,u1,u2,v=False,para=0):
        seq1=[(self.gcs[j],self.gts[j]) for j in t1]
        seq2=[(self.gcs[j],self.gts[j]) for j in t2]
        seq1_world=[self.g_world[j][0] for j in t1]
        seq2_world=[self.g_world[j][0] for j in t2]
        dist=np.zeros((len(seq1),len(seq2)))
        for u,p in enumerate(seq1):
            for w,q in enumerate(seq2):
                if p[0]==q[0]:
                    min_diff=abs(p[1]-q[1])/1000
                    if min_diff<max_diff:
                        d1=min_diff
                        pos1=np.array([seq1_world[u][key] for key in seq1_world[u].keys()])
                        # print(seq1_world[u].keys())
                        pos2=np.array([seq2_world[w][key] for key in seq2_world[w].keys()])
                        d2,k=self.dtwk(pos1,pos2)
                        dist[u,w]=u1*min_diff+u2*d2/k
                    else:
                        dist[u,w]=max_d
                else:
                    dist[u][w]=max_d
        N,M=dist.shape
        D=np.zeros(dist.shape)
        D[0,0]=dist[0,0]
        for i in range(1,N):
            D[i,0]=D[i-1,0]+dist[i,0]
        for i in range(1,M):
            D[0,i]=D[0,i-1]+dist[0,i]
        for i in range(1,N):
            for j in range(1,M):
                D[i,j]=dist[i,j]+min(D[i-1,j],min(D[i-1,j-1],D[i,j-1]))
        min_dist=D[N-1,M-1]
        # return min_dist
        i=N-1
        j=M-1
        path1=[]
        path2=[]
        k=1
        while i!=0 or j!=0:
            if i==0:
                j=j-1
            elif j==0:
                i-=1
            else:
                idx=np.argmin([D[i-1,j-1],D[i-1,j],D[i,j-1]])
                if idx==0:
                    i-=1
                    j-=1
                elif idx==1:
                    i-=1
                else:
                    j-=1
            path1.append(i)
            path2.append(j)
            k+=1
        if k==0:
            return max_d
        return min_dist/k

    def trajectory_cluster(self,idx_from_gfs,gcs,gts,threshold=10,alpha=10,beta=1):
        convert={'SQ0921':'A','SQ0922':'B','SQ0923':'C','SQ0924':'D','SQ0925':'E','SQ0926':'F','SQ0927':'G','SQ0928':'H','SQ0929':'I','SQ0930':'J','SQ0931':'K','SQ0932':'P'}
        idxs=range(len(idx_from_gfs))
        sqs=[]
        for i in idxs:
            seq=[(gcs[j],gts[j]) for j in idx_from_gfs[i]]
            sqs.append(seq)
        distmat=np.ones((len(idxs),len(idxs)))
        for i in range(len(idxs)):
            for j in range(i+1,len(idxs)):
                seq1=''
                for p in sqs[i]:
                    seq1+=convert[p[0]]
                seq2=''
                for p in sqs[j]:
                    seq2+=convert[p[0]]
                r=1-lv.ratio(seq1,seq2)
                dist=np.zeros((len(seq1),len(seq2)))
                for u,p in enumerate(sqs[i]):
                    for w,q in enumerate(sqs[j]):
                        if p[0]==q[0]:
                            min_diff=abs(p[1]-q[1])/1000.0
                            if min_diff<60:
                                dist[u][w]=min_diff
                            else:
                                dist[u][w]=max_d
                        else:
                            dist[u][w]=max_d
                ind=linear_assignment(dist)
                d=[]
                for pair in ind:
                    if dist[pair[0]][pair[1]]<max_d:
                        #print(pair[0],pair[1],dist[pair[0]][pair[1]])
                        d.append(dist[pair[0]][pair[1]])
                if len(d)==0 :
                    d=max_d
                else:
                    d=sum(d)/len(d)
                distmat[i,j]=alpha*r+beta*d
        distmat+=distmat.transpose()
        link = linkage(distmat, "average")
        clusters = fcluster(link,threshold, criterion='distance')
        new_idxs=defaultdict(list)
        group_keys={}
        for i,c in enumerate(clusters):
            new_idxs[c-1].append(i)
            group_keys[i]=c-1
        return group_keys,new_idxs
    
    
class FClusterTrajectoryGeneration:
    def __init__(self,qcs,qfs,qls,qts,tfs,tidxs,tcs,tls,gts,tpath2index,idx2pathidx,compare,dim,adj2path,parameters):
        # super(LocalCRFClusterTrajectoryGeneration,self).__init__(tracks,qfs,qls,qts,tfs,tidxs,tcs,tls,gts,tpath2index,test_cluster_idxs,idx2pathidx,compare,mode,dim,cluster_per_camera_time,camidx2tidx,**kwargs) 
        self.adj2path_factory={'graph':self.adj2path_graph,'rnmf':self.adj2path_rnmf,'fcluster':self.adj2path_fcluster}
        self.adj2path=self.adj2path_factory[adj2path]
        self.compare=compare
        # self.tracks=tracks
        self.qcs=qcs
        self.qfs=qfs
        self.qls=qls
        self.tpath2index=tpath2index
        self.idx2pathidx=idx2pathidx
        self.gts=gts
        self.gfs=tfs
        self.tls=tls
        self.tcs=tcs
        self.tidxs=tidxs
        self.dim=dim
        # self.args=kwargs
        # self.camidx2tidx=camidx2tidx
        self.qts=qts
        self.qts2=[k for k in qts]
        self.gts2=[k for k in gts]
        self.test_cluster_idxs=defaultdict(list)
        for i in range(len(self.tcs)):
            self.test_cluster_idxs[self.tls[i]].append(i)
        self.parameters=parameters
            
    def rnmf(self,S,Max_Iter=200,alpha=1,check_point=100,thresh=9*1e-3,thresh2=5e-5):
        [eig_value,_] = np.linalg.eig(S)                                                                               
        K = np.sum(np.abs(eig_value)>=thresh)                                                                                  
        K = max(K,1)                                                                                                   
        N=S.shape[0]
        I1 = np.ones([K,1])                                                                                                
        I2 = np.ones([N,1]) 
        A = np.random.rand(N, K)
        A = np.mat(A)
        for iter in range(1,Max_Iter):                                                                                     
            top = 4 * S* A + 2*alpha * I2 * I1.T + 1e-3                                                                    
            bot = 4 * A * A.T* A + 2 * alpha * A * I1 * I1.T + 1e-3                                                        
            A = np.multiply(A, np.sqrt(top/bot))                                                                           
        H = np.zeros(A.shape) 
        temp_A = A.copy()                                                                                              
        inds=[]
        for i in range(A.shape[0]):                                                                                             
            j = np.argmax(temp_A[i,:]) 
            H[i,j] = 1
        for i in range(H.shape[1]):
            args=np.where(H[:,i]==1)[0]
            if args.shape[0]==0:
                continue
            inds.append(args)
        return inds
    def adj2path_rnmf(self,A,sub_t,sub_items,threshold,para,thresh=0.5):
        S=A.copy()
        S+=S.transpose()
        S/=2
        for i in range(S.shape[0]):
            S[i,i]=1
        S[np.where(S<threshold)]=0
        paths=self.rnmf(S,Max_Iter=1+2*100,thresh=thresh,alpha=4)
        gt_path=[]
        np.random.seed(3)
        for path in paths:
            ap=[]
            for i in path:
                ap.append(sub_items[int(i)])
            gt_path.append(ap)
        return gt_path
    
    def adj2path_graph(self,adj_t,sub_t,sub_items,threshold,para):
        start=[]
        end=[]
        den=[]
        adj=np.zeros(adj_t.shape)
        for i in range(len(sub_t)):
            for j in range(i+1,len(sub_t)):
                if adj_t[i][j]>=threshold:
                    adj[i][j]=1
                    adj[j][i]=-1
        for i in range(len(sub_t)):
            s=True
            e=True
            for j in range(len(sub_t)):
                if adj[j][i]==1:
                    s=False
                if adj[i][j]==1:
                    e=False
            if s and e:
                den.append(i)
            elif s:
                start.append(i)
            elif e:
                end.append(i)
        paths=[]
        vis=np.zeros(len(sub_t))
        for i in range(len(sub_t)):
            path=[]
            if not vis[i] and (i in start or i in den):
                self.GR2(paths,path,vis,i,adj,start,end,den)
        gt_path=[]
        for path in paths:
            ap=[]
            for i in path:
                ap.append(sub_items[i])
            gt_path.append(ap)
        return gt_path
    
    def GR2(self,paths,path,vis,i,adj,start,end,den):
        vis[i]=True
        path.append(i)
        if i in end:
            t=[j for j in path]
            path.remove(i)
            paths.append(t)
            return 
        if i in den:
            paths.append([i])
            return 
        for j in range(len(vis)):
            if adj[i][j]==1:
                self.GR2(paths,path,vis,j,adj,start,end,den)
        path.remove(i)
        return 
    
    def adj2path_fcluster(self,S,sub_t,sub_items,threshold,para):
        gt_path=[]
        if S.shape[0]==1:
            gt_path.append(sub_items)
            return gt_path
        link = linkage(1-S, "average")
        clusters = fcluster(link,threshold, criterion='distance')
        paths=defaultdict(list)
        for i,c in enumerate(clusters):
            paths[c-1].append(sub_items[i])
        for key in paths.keys():
            gt_path.append(paths[key])
        return gt_path
    
    def generate_cluster_idxs(self,**kwargs):
        threshold=kwargs['threshold']
        dist=cdist(self.qfs,self.gfs,metric='cosine')
        args=np.argsort(dist,axis=1)
        distmat=pdist(self.gfs,metric='cosine')
        link = linkage(distmat,method='average')
        #link = linkage(distmat, "single")
        #link = linkage(distmat, "complete")
        #link = linkage(distmat, "weighted")
        #link = linkage(distmat, "ward")
        #link = linkage(distmat, "centroid")
        cluster_assignments = fcluster(link,threshold, criterion='distance')
        inds=get_cluster_indices(cluster_assignments)
        return args,inds
    
    #pre_info=self.preprogress(self.gfs,self.tls,self.tcs,self.tidxs,inds,self.dim,para=para)
    def preprogress(self,gts,gfs,tls,tcs,tidxs,test_cluster_idxs,dim,para=0):
        t=[]
        c=[]
        v=[]
        l=[]
        f=np.zeros((len(self.test_cluster_idxs),dim))
        ind=[]
        items=[]
        for j,key in enumerate(self.test_cluster_idxs.keys()):
            sub=[]
            sub_t=[]
            sub_c=[]
            sub_v=[]
            sub_l=[]
            sub_ind=[]
            sub_item=[]
            sub_f=np.zeros(dim)
            for i,item in enumerate(self.test_cluster_idxs[key]):
                # print(type(key))
                # print(key,item,type(tidxs))
                # print(tidxs)
                # 1/0
                cam=tcs[item]
                idx=tidxs[item]
                gf=gfs[item]
                ti=gts[item] #+np.random.randint(6000*para)*1000
                sub_f+=gf
                sub_c.append(cam)
                sub_t.append(ti)
                sub_l.append(tls[item])
                sub_ind.append(tidxs)
                sub_item.append(item)
            items.append(sub_item)
            t.append(sub_t)
            c.append(sub_c)
            l.append(sub_l)
            ind.append(sub_ind)
            f[j]=sub_f/len(self.test_cluster_idxs[key])
        items_length=np.mean([len(item) for item in items])
        return c,l,t,f,items,items_length
    
    def rankscore(self,dist,info,indices=None):
        if indices is None:
            score,loss2=rankscore(dist,self.qls,self.idx2pathidx,self.tpath2index,info)
        else:
            score,loss2=rankscore(dist,self.qls,self.idx2pathidx,self.tpath2index,info,indices=indices)
        # print('loss2',score,loss2)
        loss3=tds2(dist,self.qls,self.idx2pathidx,self.tpath2index,info,indices=indices,topk=10)
        loss4=average_search_time(dist,self.qls,self.idx2pathidx,self.tpath2index,info,indices=indices,topk=10)
        loss5=tc(dist,self.qls,self.tls,self.idx2pathidx,self.tpath2index,info,indices=indices,topk=10)
        print('TDS:',loss3)
        # print('ast',loss4)
        print('TC:',loss5)
        print('TAS',loss3*loss5)
        return score   
    
    
    def trajectory_reranking_dist(self,args,dist2,args2,idx_from_gfs,info,topk=20):
        dist=[]
        for i in range(dist2.shape[0]):
            args3=args2[i]
            sub_idx_from_gfs=[idx_from_gfs[j] for j in args3]
            temp=[]
            idxs=[]
            for item in sub_idx_from_gfs:
                sub_f=np.array([self.gfs[j] for j in item])
                dist4=cdist(self.qfs[i][np.newaxis],sub_f,metric='cosine')
                args4=np.argsort(dist4)[0]
                temp.extend(dist4[0,:].tolist())
                idxs.extend([item[j] for j in args4])
            idxs2=[]
            temp2=[]
            for j,item in enumerate(idxs):
                if item not in idxs2:
                    idxs2.append(item)
                    temp2.append(temp[j])
            dist.append(temp2)
        return dist
    
    def trajectory_reranking(self,args,dist2,args2,idx_from_gfs,info,topk=20):
            #Trajectory Reranking
        for i in range(dist2.shape[0]):
            args3=args2[i]
            sub_idx_from_gfs=[idx_from_gfs[j] for j in args3]
            idxs=[]
            for item in sub_idx_from_gfs:
                sub_f=np.array([self.gfs[j] for j in item])
                dist4=cdist(self.qfs[i][np.newaxis],sub_f,metric='cosine')
                args4=np.argsort(dist4)[0]
                idxs.extend([item[j] for j in args4])
            idxs2=[]
            for item in idxs:
                if item not in idxs2:
                    idxs2.append(item)
            args[i][:len(idxs2)]=np.array(idxs2).astype(np.int64)
        return args

    
    def forward(self,parameters=None,threshold=0.1,re_rank=False,k1=6,k2=2,lambda_value=0.8,alpha1=5,lambda1=1,lambda2=1,alpha2=11,topk=20,st_topk=20,func=None,para=0,seed=3):
        
        np.random.seed(seed)
        start=time.perf_counter()
        args,inds=self.generate_cluster_idxs(threshold=threshold,topk=topk,para=para)
        pre_info=self.preprogress(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,inds,self.dim,para=para)
        dist=cdist(self.qfs,pre_info[3],metric='cosine')
        pre_score=self.rankscore(dist,pre_info[4])
        pre_length=pre_info[5]
        after_info=self.generate_path(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,inds,self.idx2pathidx,self.tpath2index,self.compare,self.dim,para)
    
        score_cluster_length=after_info[2]
        idx_from_gfs=after_info[1]
        qfs=norm(torch.Tensor(self.qfs))
        if re_rank:
            tfs=norm(torch.Tensor(after_info[0]))
            q_g_dist = np.dot(qfs, np.transpose(tfs))
            q_q_dist = np.dot(qfs, np.transpose(qfs))
            g_g_dist = np.dot(tfs, np.transpose(tfs))
            dist2 = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1,k2=k2, lambda_value=lambda_value)
        else:
            dist2=cdist(qfs,after_info[0],metric='cosine')
        indices = np.argsort(dist2, axis=1)
        num_q=dist2.shape[0]
        num_g=dist2.shape[1]
        # if st_rerank:
        #     interval=20
        #     if osp.exists('hist.pkl') :
        #         hist=pkl.load(open('hist.pkl','rb'),encoding='latin')
        #     else:
        #         hist=func(o=10*6,interval=interval)
        #         with open('hist.pkl','wb') as out:
        #             pkl.dump(hist,out)
        #     for i in range(num_q):
        #         score_st = np.zeros(st_topk)
        #         argst=indices[i,:st_topk]
        #         score=dist2[i,argst]
        #         for w,j in enumerate(argst):
        #             for k in idx_from_gfs[j]:
        #                 score_st[w]=max(score_st[w],self.compare(self.qcs[i],self.qts[i],self.tcs[k],self.gts[k],prob=True))
        #         score_st= 1/(1+lambda1*np.exp(-alpha1*score))/(1+lambda2*np.exp(alpha2*score_st))
        #         indices[i,:topk]=argst[np.argsort(score_st)]
        #     score_cluster=self.rankscore(dist2,after_info[1],indices=indices)
        # else:
        score_cluster=self.rankscore(dist2,after_info[1])
        args2=indices.astype(np.int64)
        args=self.trajectory_reranking(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        dist3=self.trajectory_reranking_dist(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        end=time.perf_counter()
        with open('tr_distmat.pkl','wb') as out:
            pkl.dump({'dist2':dist3,'args':args,'idx_from_gfs':idx_from_gfs,'dist':dist2,'args2':args2,'idx2pathidx':self.idx2pathidx,'tpath2index':self.tpath2index},out)
        return pre_score,pre_length,score_cluster,score_cluster_length,args #,idx_from_gfs,args2
        
    def crf(self,adj,T=10,u=0.1,u1=1.7,u2=2.0,alpha=0.6,nc_threshold=0.6,v=1):
        adj_t=np.array(adj).copy()
        g=dict()
        v=1
        for st in range(T):                     
            new_a=np.zeros_like(adj_t)                                                                      
            for i in range(new_a.shape[0]):                  
                if i in g.keys():                                                   
                    idxs=g[i]                                                                           
                else:                                                                                
                    idxs=[]
                for j in range(new_a.shape[0]):                        
                    f1=adj_t[i,:]/np.linalg.norm(adj_t[i,:])                                                
                    f2=adj_t[j,:]/np.linalg.norm(adj_t[j,:])                             
                    sim=np.sum(f1*f2)                                                                      
                    e=u1*adj_t[i,j]+u2*sim
                    if len(idxs)!=0:                                                                    
                        w=0                                                                             
                        for idx in idxs:                                                                
                            w+=b[idx][j]                                                            
                        w/=len(idxs)    
                        e= e + v*(1-2*w)
                    new_a[i,j]=np.exp(alpha*e)         
                new_a[i,:]/=new_a[i,i] 
            adj_t=new_a
        return adj_t

    def adj2path_graph(self,adj_t,sub_t,sub_items,threshold,para):
        start=[]
        end=[]
        den=[]
        adj=np.zeros(adj_t.shape)
        for i in range(len(sub_t)):
            for j in range(i+1,len(sub_t)):
                if adj_t[i][j]>=threshold:
                    adj[i][j]=1
                    adj[j][i]=-1
        for i in range(len(sub_t)):
            s=True
            e=True
            for j in range(len(sub_t)):
                if adj[j][i]==1:
                    s=False
                if adj[i][j]==1:
                    e=False
            if s and e:
                den.append(i)
            elif s:
                start.append(i)
            elif e:
                end.append(i)
        paths=[]
        vis=np.zeros(len(sub_t))
        for i in range(len(sub_t)):
            path=[]
            if not vis[i] and (i in start or i in den):
                self.GR2(paths,path,vis,i,adj,start,end,den)
        gt_path=[]
        for path in paths:
            ap=[]
            for i in path:
                ap.append(sub_items[i])
            gt_path.append(ap)
        return gt_path
# 
    # def generate_path(self,tracks,gts,gfs,tls,tcs,tidxs,test_cluster_idxs,idx2pathidx,tpath2index,compare,mode,dim,para):
    def generate_path(self,gts,gfs,tls,tcs,tidxs,test_cluster_idxs,idx2pathidx,tpath2index,compare,dim,para):
                     
        t=[]
        c=[]
        v=[]
        l=[]
        f=np.zeros((len(test_cluster_idxs),dim))
        ind=[]
        items=[]
        gt_path=defaultdict(list)
        total_items=[]
        for k,key in enumerate(test_cluster_idxs.keys()):
            sub=[]
            sub_t=[]
            sub_c=[]
            sub_v=[]
            sub_l=[]
            sub_item=[]
            sub_f=np.zeros(dim)
            for i,item in enumerate(test_cluster_idxs[key]):
                cam=tcs[item]
                idx=tidxs[item]
                # ti=get_tracks_time(tracks,cam,idx,mode=mode)
                sub_c.append(cam)
                # sub_t.append(ti)
                sub_l.append(tls[item])
                sub_item.append(item)
            # t.append(sub_t)
            c.append(sub_c)
            l.append(sub_l)
            num=len(sub_c)
            items=test_cluster_idxs[key]
            sub_items=items
            gt_path[key].append(sub_items)
        length=np.mean([len(test_cluster_idxs[key]) for key in test_cluster_idxs.keys()])
        test_feature=[]
        idx_from_gfs=[]
        global_f=defaultdict(list)
        global_idx_from_gfs=defaultdict(list)
        i=0
        #key2 cluster_key
        for key2 in gt_path.keys():
            args=gt_path[key2]
            sub_feature=[]
            sub_idx_from_gfs=[]
            for j in range(len(args)):
                sub_f=np.array([gfs[k] for k in args[j]])
                idx_from_gfs.append([k for k in args[j]])
                sub_feature.append(np.mean(sub_f,axis=0))
                sub_idx_from_gfs.append([k for k in args[j]])
                if len(test_feature)==0:
                    test_feature=np.mean(sub_f,axis=0)[np.newaxis]
                else:
                    test_feature=np.vstack((test_feature,np.mean(sub_f,axis=0)))
                i+=1
            global_f[key2].extend(sub_feature)
            global_idx_from_gfs[key2].extend(sub_idx_from_gfs)
        return test_feature,idx_from_gfs,length,global_f,global_idx_from_gfs
    
    def forward_trajectory(self,threshold=0.1,re_rank=False,st_rerank=False,k1=6,k2=2,lambda_value=0.8,alpha1=1,lambda1=1,lambda2=1,topk=20,st_topk=20,func=None,para=0,seed=3):
        np.random.seed(seed)
        args,inds=self.generate_cluster_idxs(threshold=threshold,topk=topk,para=para)
        pre_info=self.preprogress(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,inds,self.dim)
        dist=cdist(self.qfs,pre_info[3],metric='cosine')
        #pre_score=self.rankscore(dist,pre_info[4])
        pre_length=pre_info[5]
        after_info=self.generate_path(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,inds,self.idx2pathidx,self.tpath2index,self.compare,self.dim,para)
                                     
        score_cluster_length=after_info[2]
        idx_from_gfs=after_info[1]
        qfs=norm(torch.Tensor(self.qfs))
        tfs=norm(torch.Tensor(after_info[0]))
        dist2=cdist(qfs,tfs,metric='cosine')
        indices = np.argsort(dist2, axis=1)
        args2=indices.astype(np.int64)
        args=self.trajectory_reranking(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        dist3=self.trajectory_reranking_dist(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        with open('tr_distmat.pkl','wb') as out:
            pkl.dump({'dist2':dist3,'args':args,'idx_from_gfs':idx_from_gfs,'dist':dist2,'args2':args2,'idx2pathidx':self.idx2pathidx,'tpath2index':self.tpath2index},out)
        return qfs,tfs,idx_from_gfs,args 

def print_log(text):
    print(text)
def fcluster_main(config):
    #load spatio-temporal model
    th=0.9+0.01*5
    name2idx,idx2name,lm=loadlocation(config['spatiotemporal']['locationroot'])
    compare=generate_compare('UM',config['spatiotemporal']['name'],lm,name2idx,config['spatiotemporal']['modelroot'],convertname=True,thresh=th)
    if sys.version[0]=='3':
        model=pkl.load(open(osp.join(config['spatiotemporal']['modelroot'],config['spatiotemporal']['name']+'.pkl'),'rb'),encoding='latin')
    else:
        model=pkl.load(open(osp.join(config['spatiotemporal']['modelroot'],config['spatiotemporal']['name']+'.pkl'),'rb'))
    compare.set_model(model)
    with open(config['dataset']['pos_dataset'],'rb') as infile:                                                                          
        g_world=pkl.load(infile,encoding='bytes')   
    #load dataset
    with open(config['dataset']['path'],'rb') as infile:
        datas=pkl.load(infile,encoding='latin')
    # print(datas.keys())
    qfs=datas['qfs']
    qls=datas['qls']
    qcs=datas['qcs']
    qts=datas['qts']
    tfs=datas['tfs']
    tidxs=datas['tidxs']
    tls=datas['tls']
    gts=datas['gts']
    tcs=datas['tcs']
    # print(len(tcs),len(tidxs))
    # 1/0
    tpath2index=datas['tpath2index']
    idx2pathidx=datas['idx2pathidx']
    st_rerank=config['st_rerank']
    adj2path=config['adj2path']
    dim=config['dim']
    group_pidxs2gidxs={}
    group_pidxs_dict=pkl.load(open(config['dataset']['group_dataset'],'rb'))  
    remove_pidxs=[]
    group_ids=[]
    t=0
    for key in group_pidxs_dict.keys():
        group_pidxs=group_pidxs_dict[key]
        flag=False
        for idx in group_pidxs:
            if idx in group_pidxs2gidxs.keys():
                break
            else:
                group_pidxs2gidxs[idx]=t
                flag=True
        if flag:
            t+=1
    #print(group_pidxs2gidxs)
    qls2=[]
    t+=1
    for ql in qls:
        if ql in group_pidxs2gidxs.keys():
            qls2.append(group_pidxs2gidxs[ql])
        else:
            qls2.append(t)
            t+=1
        
    FCTG=FClusterTrajectoryGeneration(qcs,qfs,qls,qts,tfs,tidxs,tcs,tls,gts,tpath2index,idx2pathidx,compare,dim,adj2path,config['fcluster'])
    GSMBaseline=GroupSearchBaseMethod(tcs,gts,qcs,qts,g_world,compare)
    for i in range(1,100):
        # score_gt,score_gt_length,score_cluster,score_cluster_length,args=LCTG.forward(threshold=config['cluster_threshold'],re_rank=config['rerank']['enable'],k1=config['rerank']['k1'],k2=config['rerank']['k2'],lambda_value=config['rerank']['lambda'])
        #test
        qfs2,tfs2,idx_from_gfs,args=FCTG.forward_trajectory(threshold=0.01*2,seed=8,para=0)
        dist = np.zeros((len(qfs), len(tfs)))
        ranks=[1,3,5,10]
        cmc, mAP = evaluate_args(args, dist, qls, tls, qcs, tcs)
        print_log("PTR+ Results ----------")
        print_log("[{}] mAP: {:.1%}".format(i,mAP))
        print_log("CMC curve")
        for r in ranks:
            print_log("[{}] Rank-{:<3}: {:.1%}".format(i,r, cmc[r-1]))
        print_log("PTR Results ----------") 
        gls=[]
        for path in idx_from_gfs:
            gls.append([tls[j] for j in path])
        dist2=cdist(qfs2,tfs2,metric='cosine')
        args3=np.argsort(dist2,axis=1)
        m=eval_PTR_map(args3,qls,gls)
        print_log("PTR mAP: {:.1%}".format(m))
        FCTG.rankscore(dist2,idx_from_gfs)
        # distmat,args=GSMBaseline.forward(qfs2,tfs2,idx_from_gfs,config['group_detection'],topk=config['group_detection']['topk'],para=i)
        distmat,args=GSMBaseline.forward(qfs2,tfs2,idx_from_gfs,config['group_detection'],topk=1+2,para=i)
        pr,re,f1,t= evaluate_group_search(distmat,qls2,threshold=10+i)
        print_log("[{}]Precision: {:.1%}".format(i,pr))
        print_log("[{}]Recall: {:.1%}".format(i,re))
        print_log("[{}]F1-Score: {:.1%}".format(i,f1))
        print_log("[{}]T: {}".format(i,t))
        print_log("------------------")
        with open('distmat.pkl','wb') as out:
            pkl.dump({'distmat':distmat,'gqls':qls2},out)
    return 