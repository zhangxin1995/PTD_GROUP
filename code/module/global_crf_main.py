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
import Levenshtein as lv
from module.re_ranking import re_ranking
from sklearn.utils.linear_assignment_ import linear_assignment
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

def activation(x,a=1):
    return x
    # return np.tanh(a*x)

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
        max_d=10*8 #self.config['max_d'] #10*7
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
    
class GlobalCRFClusterTrajectoryGeneration:
    def __init__(self,qcs,qfs,qls,qts,tfs,tidxs,tcs,tls,gts,tpath2index,idx2pathidx,compare,dim,adj2path,parameters):
        # super(LocalCRFClusterTrajectoryGeneration,self).__init__(tracks,qfs,qls,qts,tfs,tidxs,tcs,tls,gts,tpath2index,test_cluster_idxs,idx2pathidx,compare,mode,dim,cluster_per_camera_time,camidx2tidx,**kwargs) 
        self.adj2path_factory={'fcluster':self.adj2path_fcluster}
        self.adj2path=self.adj2path_factory['fcluster']
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
        self.qts=qts
        self.qts2=[k for k in qts]
        self.gts2=[k for k in gts]
        self.test_cluster_idxs=defaultdict(list)
        for i in range(len(self.tcs)):
            self.test_cluster_idxs[self.tls[i]].append(i)
        self.parameters=parameters
        
    def adj2path_fcluster(self,S,sub_t,sub_items,threshold,para):
        gt_path=[]
        S[S>1]=1
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
            gls=[]
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

    def generate_path(self,gts,gfs,tls,tcs,tidxs,test_cluster_idxs,idx2pathidx,tpath2index,compare,dim,para):
        print('para:',para)
        t=[]
        c=[]
        v=[]
        l=[]
        f=np.zeros((len(test_cluster_idxs),dim))
        ind=[]
        items=[]
        total_items=[]
        gt_path=defaultdict(list)
        #np.random.seed(int(time.time()))
        for k,key in enumerate(test_cluster_idxs.keys()):
            sub=[]
            sub_t=[]
            sub_c=[]
            sub_v=[]
            sub_l=[]
            sub_item=[]
            for i,item in enumerate(test_cluster_idxs[key]):
                cam=tcs[item]
                idx=tidxs[item]
                ti=gts[item] #+np.random.randint(600*para)*1000
                sub_c.append(cam)
                sub_t.append(ti)
                sub_l.append(tls[item])
                sub_item.append(item)
            t.append(sub_t)
            c.append(sub_c)
            l.append(sub_l)
            num=len(sub_c)
            rest=np.argsort([np.min(item) for item in sub_t])
            items=test_cluster_idxs[key]
            sub_t = [sub_t[i] for i in rest]
            sub_c=[sub_c[i] for i in rest]
            sub_items=[items[i] for i in rest]
            sub_f=np.array([gfs[k] for k in sub_items])
            adj=np.zeros((len(sub_t),len(sub_t)))
            fuzhu=np.zeros((len(sub_t),len(sub_t)))
            td=np.zeros((len(sub_t),len(sub_t)))
            adj_t=np.zeros((len(sub_t),len(sub_t)))
            dist_sub_f=cdist(sub_f,sub_f,metric='cosine')
            for i in range(len(sub_t)):
                adj_t[i,i]=1
                for j in range(i+1,len(sub_t)):
                    fuzhu[i][j]=compare(sub_c[i],sub_t[i],sub_c[j],sub_t[j],prob=True)
                    td[i][j]=sub_t[i]-sub_t[j]
                    td[j][i]=sub_t[i]-sub_t[j]
                    if compare(sub_c[i],sub_t[i],sub_c[j],sub_t[j]):
                        adj[i][j]=1
                        adj[j][i]=-1
                    adj_t[i][j]=compare(sub_c[i],sub_t[i],sub_c[j],sub_t[j],prob=True)#*1.0/(1+0.01*para*np.exp(0.5*dist_sub_f[i][j]))
                    adj_t[j][i]=adj_t[i][j]
            alpha=self.parameters['alpha']
            u=self.parameters['u1']
            T=self.parameters['T']
            nc_threshold=self.parameters['nc_threshold']
            u1=self.parameters['u1']
            u2=self.parameters['u2']
            threshold=self.parameters['threshold']
            adj_t=self.crf(adj_t,T=T,u=u,alpha=alpha,nc_threshold=nc_threshold,u1=u1,u2=u2)
            gt_path[key]=self.adj2path(adj_t,sub_t,sub_items,threshold,para)
        length=np.mean([len(test_cluster_idxs[key]) for key in test_cluster_idxs.keys()])
        test_feature=[] #np.zeros((0,dim))
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
                    t=np.mean(sub_f,axis=0)[np.newaxis]
                    test_feature=np.vstack((test_feature,t))
                i+=1
            global_f[key2].extend(sub_feature)
            global_idx_from_gfs[key2].extend(sub_idx_from_gfs)
        return test_feature,idx_from_gfs,length,global_f,global_idx_from_gfs
    
    def global_forward(self,threshold=0,re_rank=False,st_rerank=False,k1=6,k2=2,lambda_value=0.8,alpha1=5,lambda1=1,lambda2=1,alpha2=11,topk=20,st_topk=20,func=None,para=0,seed=3): 
        np.random.seed(seed)
        args,inds=self.generate_cluster_idxs(threshold=0,topk=topk,para=para)
        pre_info=self.preprogress(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,inds,self.dim,para=para)
        dist=cdist(self.qfs,pre_info[3],metric='cosine')
        pre_score=self.rankscore(dist,pre_info[4])
        pre_length=pre_info[5]
        after_info=self.generate_path_RS_crf(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,self.idx2pathidx,self.tpath2index,self.compare,self.dim,para)
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
        if st_rerank:
            #np.random.seed(int(time.time()))
            interval=20
            if osp.exists('hist.pkl') :
                hist=pkl.load(open('hist.pkl','rb'))
            else:
                hist=func(o=10*6,interval=interval)
                with open('hist.pkl','wb') as out:
                    pkl.dump(hist,out)
            for i in range(num_q):
                score_st = np.zeros(st_topk)
                argst=indices[i,:st_topk]
                score=dist2[i,argst]
                for w,j in enumerate(argst):
                    for k in idx_from_gfs[j]:
                        ct=self.compare(self.qcs[i],self.qts[i],self.tcs[k],self.gts[k],prob=True)
                        if ct<para*0.1:
                            ct=0
                        score_st[w]=max(score_st[w],ct)
                score_st= 1/(1+lambda1*np.exp(-alpha1*score))/(1+lambda2*np.exp(alpha2*score_st))
                indices[i,:topk]=argst[np.argsort(score_st)]
            score_cluster=self.rankscore(dist2,after_info[1],indices=indices)
        else:
            score_cluster=self.rankscore(dist2,after_info[1])
        args2=indices.astype(np.int64)
        print(3,args.shape)
        args=self.trajectory_reranking(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        print(3,args.shape)
        print(args)
        dist3=self.trajectory_reranking_dist(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        with open('tr_distmat.pkl','wb') as out:
            pkl.dump({'dist2':dist3,'args':args,'idx_from_gfs':idx_from_gfs,'dist':dist2,'args2':args2,'idx2pathidx':self.idx2pathidx,'tpath2index':self.tpath2index},out)
        return pre_score,pre_length,score_cluster,score_cluster_length,args #,idx_from_gfs,args2
    
    def global_crf(self,adj_f,adj_t,T=10,T1=1,T2=1,u11=1.7,u12=2.0,alpha1=0.6,u21=1.7,u22=2.0,alpha2=0.6,u4=0,nc_threshold1=0.6,nc_threshold2=0.6,v=1,t1=0.16,t2=0.16,t3=0.16,t4=0.16,b1=0,b2=0,nc_threshold=0.5,para=0,above_threshold1=0.8,above_threshold2=0.8, global_group_idxs=None,idx2global_group_id=None):
        def update_A_by_B(A,B=None,T=10,u=0.1,u1=1.7,u2=2.0,alpha=0.6,nc_threshold=0.6,v=1,para=0,t1=0.16,t2=0.16,u4=0.,above_threshold=0,global_group_idxs=None,idx2global_group_id=None):
            #print(idx2global_group_id)
            T2=1
            T1=1
            B=np.array(B).copy()
            A=np.array(A).copy()
            B[B<nc_threshold]=0
            print('ah',above_threshold1)
            if B is None:
                B=np.ones(A.shape)
            for st in range(T):                     
                new_A=np.zeros_like(A)
                group_values=[]
                if idx2global_group_id is not None:
                    for i in range(new_A.shape[0]):
                        # print(new_A.shape,len(idx2global_group_id))
                        gid=idx2global_group_id[i]
                        g=global_group_idxs[gid]
                        for i in range(new_A.shape[0]):
                            if  len(g)!=0: 
                                pairs=[]
                                for k in g:
                                    if k==i:
                                        continue
                                    else:
                                        pairs.append([i,k])
                                sim=np.zeros(A[0,:].shape)
                                for i,pair in enumerate(pairs):
                                    idx1=pair[0]
                                    idx2=pair[1]
                                    f1=A[idx1,:]
                                    f2=A[idx2,:]
                                    idx1=np.where(f1>above_threshold)[0]
                                    idx2=np.where(f2>above_threshold)[0]
                                    idx=np.unique(np.hstack((idx1,idx2)))
                                    sim[idx]+=f1[idx]*f2[idx]
                                if len(pairs)!=0:
                                    group_values.append(sim/len(pairs))
                                else:
                                    group_values.append(sim)
                for i in range(new_A.shape[0]):            
                    for j in range(new_A.shape[0]):                        
                        idx1=np.where(A[i,:]>above_threshold)[0]
                        idx2=np.where(A[j,:]>above_threshold)[0]
                        idx=np.unique(np.hstack((idx1,idx2)))
                        f1=activation(A[i,idx],a=t1)*activation(B[i,idx],a=t2)
                        f2=activation(A[j,idx],a=t1)*activation(B[j,idx],a=t2)
                        #f1=activation(A[i,:],a=t1)*activation(B[i,:],a=t2)
                        #f2=activation(A[j,:],a=t1)*activation(B[j,:],a=t2)
                        sim=np.sum(f1*f2)/(np.linalg.norm(f1)*np.linalg.norm(f2))                        
                        e=u1*A[i,j]*B[i,j]+u2*sim
                        new_A[i,j]=e   
                    if idx2global_group_id is not None:
                        new_A[i,:]+=u4*group_values[i]
                    new_A[i,]=np.exp(alpha*new_A[i,:])         
                    new_A[i,:]/=new_A[i,i]
                A=new_A
            return A
        for iter1 in range(T):
            for i in range(T1):
                adj_t2=update_A_by_B(adj_t,B=adj_f,T=T2,u1=u21,u2=u22,alpha=alpha2,nc_threshold=nc_threshold1,para=para,t1=t1,t2=t2,u4=u4,above_threshold=above_threshold1,global_group_idxs=global_group_idxs,idx2global_group_id=idx2global_group_id)
                adj_t=adj_t*b1+adj_t2*(1-b1)
            for i in range(T2):
                adj_f2=update_A_by_B(adj_f,B=adj_t,T=T1,u1=u11,u2=u12,alpha=alpha1,para=para,nc_threshold=nc_threshold2,t1=t3,t2=t4,above_threshold=above_threshold2)
                adj_f=adj_f*b2+adj_f2*(1-b2)
        print(np.where(adj_f<0))
        # exit()
        return adj_f,adj_t 
    
    def cluster_per_camera_time(self,tcs,tidxs,gts,threshold=30):
        camera_group2idx=defaultdict(list)
        camera_group_t=defaultdict(list)
        for i in range(len(tcs)):
            camera_group2idx[tcs[i]].append(i)
            camera_group_t[tcs[i]].append([gts[i]/1000.0])
        total_group_id=0
        idx2global_group_id={}
        global_group_idxs=[]
        per_cam_group=[]
        for cam in camera_group2idx.keys():
            ts=camera_group_t[cam]
            k=0
            if len(ts)>1:
                dist=cdist(ts,ts,metric='euclidean')
                Z=linkage(dist,method='average')
                cluster_assignments = fcluster(Z, threshold, criterion='distance')
                res=get_cluster_indices(cluster_assignments)
                for i in res.keys():
                    idxs=res[i]
                    if len(idxs)>1:
                        k+=1
                    global_group_idxs.append([camera_group2idx[cam][j] for j in idxs])
                    for idx in idxs:
                        idx2global_group_id[camera_group2idx[cam][idx]]=len(global_group_idxs)-1
            else:
                global_group_idxs.append([camera_group2idx[cam][0]])
                idx2global_group_id[camera_group2idx[cam][0]]=len(global_group_idxs)-1
            per_cam_group.append(k)
        print(per_cam_group,sum(per_cam_group))
        return global_group_idxs,idx2global_group_id
    
                
    def generate_path_RS_crf(self,gts,gfs,tls,tcs,tidxs,idx2pathidx,tpath2index,compare,dim,para):
        print('para:',para)
        global_group_idxs,idx2global_group_id=self.cluster_per_camera_time(tcs,tidxs,gts,threshold=10*6)
        t=[]
        c=[]
        v=[]
        l=[]
        ind=[]
        items=[]
        total_items=[]
        gt_path=defaultdict(list)
        sub=[]
        sub_t=[]
        sub_c=[]
        sub_v=[]
        sub_l=[]
        sub_item=[]
        for item in range(len(tcs)):
            cam=tcs[item]
            idx=tidxs[item]
            ti=gts[item] #+np.random.randint(600*para)*1000
            sub_t.append(ti)
        items=range(len(tcs))
        td=np.zeros((len(sub_t),len(sub_t)))
        adj_t=np.zeros((len(sub_t),len(sub_t)))
        adj_f=1-cdist(self.gfs,self.gfs,metric='cosine')
        for i in range(len(sub_t)):
            adj_t[i,i]=1
            for j in range(i+1,len(sub_t)):
                td[i][j]=sub_t[i]-sub_t[j]
                td[j][i]=sub_t[i]-sub_t[j]
                adj_t[i][j]=compare(tcs[i],sub_t[i],tcs[j],sub_t[j],prob=True)#*1.0/(1+0.01*para*np.exp(0.5*dist_sub_f[i][j]))
                adj_t[j][i]=adj_t[i][j]

        #GROUP
        alpha1=float(self.parameters['alpha1'])
        alpha2=float(self.parameters['alpha2']) #0.01*5+0.1*1
        T=self.parameters['T'] #2
        T1=self.parameters['T1'] #$1
        T2=self.parameters['T2'] #1
        u11=float(self.parameters['u11']) #0.1*9+0.01*10
        u12=float(self.parameters['u12'] )#'0.1*2
        u21=float(self.parameters['u21'] )#0.35+0.01*5
        u22=float(self.parameters['u22'] )#0.1+0.01*6
        # t1=0.0000000000001*para #float(self.parameters['t1'] )
        
        threshold=0.01*3 #float(self.parameters['threshold']) #0.01*2+0.005+0.001*5 #+0.001*3
        nc_threshold1=float(self.parameters['nc_threshold1'] )#'1-0.1*2
        nc_threshold2=float(self.parameters['nc_threshold2'] )
        b1=0.1*3 #float(self.parameters['b1'] )#0.0 #1*para
        b2=float(self.parameters['b2'] )
        above_threshold1=float(self.parameters['above_threshold1']) #'1-0.1*para
        above_threshold2=float(self.parameters['above_threshold2'] )#0.1
        # u4=0.02*(para-1)
        t1=1
        t2=0.1*3
        u4=0.01*13
        adj_f,adj_t=self.global_crf(adj_f,adj_t,T=T,T1=T1,T2=T2,u11=u11,u12=u12,u21=u21,u22=u22,alpha1=alpha1,alpha2=alpha2,nc_threshold1=nc_threshold1,nc_threshold2=nc_threshold2,para=para,u4=u4,b1=b1,b2=b2,above_threshold1=above_threshold1,above_threshold2=above_threshold2,global_group_idxs=global_group_idxs,idx2global_group_id=idx2global_group_id,t1=t1,t2=t2,t3=t1,t4=t2)
        adj_f2=np.zeros(int(adj_f.shape[0]*(adj_f.shape[0]-1)/2))
        i=0
        j=0
        k=0
        while i<adj_f.shape[0]:
            for j in range(i+1,adj_f.shape[0]):
                adj_f2[k]=adj_f[i,j]
                k+=1
            i+=1
        temp_gt_path=self.adj2path_fcluster(adj_f2,sub_t,items,threshold,para)
        #temp_gt_path=self.adj2path_rnmf(adj_f,sub_t,items,threshold,para,thresh=thresh)
        #length=np.mean([len(test_cluster_idxs[key]) for key in test_cluster_idxs.keys()])
        gt_path={i:item for i,item in enumerate(temp_gt_path)}
        length=0
        test_feature=[]
        idx_from_gfs=[]
        global_f=defaultdict(list)
        global_idx_from_gfs=defaultdict(list)
        i=0
        
        for key2 in gt_path.keys():
            args=gt_path[key2]
            sub_feature=[]
            sub_idx_from_gfs=[]
            sub_f=np.array([gfs[k] for k in args])
            idx_from_gfs.append([k for k in args])
            sub_feature.append(np.mean(sub_f,axis=0))
            sub_idx_from_gfs.append([k for k in args])
            if len(test_feature)==0:
                test_feature=np.mean(sub_f,axis=0)[np.newaxis]
            else:
                t=np.mean(sub_f,axis=0)
                test_feature=np.vstack((test_feature,np.mean(sub_f,axis=0)))
            global_f[key2].extend(sub_feature)
            global_idx_from_gfs[key2].extend(sub_idx_from_gfs)
        return test_feature,idx_from_gfs,length,global_f,global_idx_from_gfs
    
    def global_forward_trajectory(self,threshold=0.1,re_rank=False,st_rerank=False,k1=6,k2=2,lambda_value=0.8,alpha1=1,lambda1=1,lambda2=1,topk=20,st_topk=20,func=None,para=0,seed=3):
        np.random.seed(seed)
        args,inds=self.generate_cluster_idxs(threshold=0.03,topk=topk,para=para)
        pre_info=self.preprogress(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,inds,self.dim)
        dist=cdist(self.qfs,pre_info[3],metric='cosine')
        pre_length=pre_info[5]
        after_info=self.generate_path_RS_crf(self.gts,self.gfs,self.tls,self.tcs,self.tidxs,self.idx2pathidx,self.tpath2index,self.compare,self.dim,para)
        score_cluster_length=after_info[2]
        idx_from_gfs=after_info[1]
        qfs=norm(torch.Tensor(self.qfs))
        tfs=norm(torch.Tensor(after_info[0]))
        dist2=cdist(qfs,tfs,metric='cosine')
        indices = np.argsort(dist2, axis=1)
        args2=indices.astype(np.int64)
        # print(1,args2.shape,tfs.size(),len(idx_from_gfs))
        # exit()
        # print(1,args)
        args=self.trajectory_reranking(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        dist3=self.trajectory_reranking_dist(args,dist2,args2,idx_from_gfs,after_info,topk=topk)
        # print(2,args)
        # print(2,args.shape)

        with open('tr_distmat.pkl','wb') as out:
            pkl.dump({'dist2':dist3,'args':args,'idx_from_gfs':idx_from_gfs,'dist':dist2,'args2':args2,'idx2pathidx':self.idx2pathidx,'tpath2index':self.tpath2index},out)
        return qfs,tfs,idx_from_gfs,args
    
def print_log(text):
    print(text)
    
    
def global_crf_main(config):
    #load spatio-temporal model
    th=0.9+0.01*5
    name2idx,idx2name,lm=loadlocation(config['spatiotemporal']['locationroot'])
    compare=generate_compare('UM',config['spatiotemporal']['name'],lm,name2idx,config['spatiotemporal']['modelroot'],convertname=True,thresh=th)
    if sys.version[0]=='3':
        model=pkl.load(open(osp.join(config['spatiotemporal']['modelroot'],config['spatiotemporal']['name']+'.pkl'),'rb'),encoding='latin')
    else:
        model=pkl.load(open(osp.join(config['spatiotemporal']['modelroot'],config['spatiotemporal']['name']+'.pkl'),'rb'))
    compare.set_model(model) 

    with open(config['dataset']['path'],'rb') as infile:
        datas=pkl.load(infile,encoding='latin')
    g_world=datas['g_world']
    qfs=datas['qfs']
    qls=datas['qls']
    qcs=datas['qcs']
    qts=datas['qts']
    tfs=datas['tfs']
    tidxs=datas['tidxs']
    tls=datas['tls']
    gts=datas['gts']
    tcs=datas['tcs']
    tpath2index=datas['tpath2index']
    idx2pathidx=datas['idx2pathidx']
    adj2path=config['adj2path']
    dim=config['dim']
    qls2=datas['qls2']
    ranks=[1,3,5,10]

    GCTG=GlobalCRFClusterTrajectoryGeneration(qcs,qfs,qls,qts,tfs,tidxs,tcs,tls,gts,tpath2index,idx2pathidx,compare,dim,adj2path,config['crf'])
    GSMBaseline=GroupSearchBaseMethod(tcs,gts,qcs,qts,g_world,compare,config['group_detection'])
    i=0 
    qfs2,tfs2,idx_from_gfs,args=GCTG.global_forward_trajectory(threshold=config['crf']['threshold'],seed=8,para=i)
    dist = np.zeros((len(qfs), len(tfs)))
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
    for i in range(1,100):  
        
        # qfs2,tfs2,idx_from_gfs,args=GCTG.global_forward_trajectory(threshold=config['crf']['threshold'],seed=8,para=i)
        # dist = np.zeros((len(qfs), len(tfs)))
        # cmc, mAP = evaluate_args(args, dist, qls, tls, qcs, tcs)
        # print_log("PTR+ Results ----------")
        # print_log("[{}] mAP: {:.1%}".format(i,mAP))
        # print_log("CMC curve")
        # for r in ranks:
        #     print_log("[{}] Rank-{:<3}: {:.1%}".format(i,r, cmc[r-1]))
        # print_log("PTR Results ----------") 
        # gls=[]
        # for path in idx_from_gfs:
        #     gls.append([tls[j] for j in path])      
        dist2=cdist(qfs2,tfs2,metric='cosine')
        args3=np.argsort(dist2,axis=1)
        m=eval_PTR_map(args3,qls,gls)
        print_log("PTR mAP: {:.1%}".format(m))
        GCTG.rankscore(dist2,idx_from_gfs)
        print_log("Results ------------------Tracklet to Group(Baseline)")
        distmat,args=GSMBaseline.forward(qfs2,tfs2,idx_from_gfs,threshold=config['group_detection']['threshold1'],topk=config['group_detection']['topk'],para=i)
        pr,re,f1,t= evaluate_group_search(distmat,qls2,threshold=config['group_detection']['threshold2'])
        print_log("[{}]Precision: {:.1%}".format(i,pr))
        print_log("[{}]Recall: {:.1%}".format(i,re))
        print_log("[{}]F1-Score: {:.1%}".format(i,f1))
        print_log("[{}]T: {}".format(i,t))
        print_log("------------------")
        with open('distmat.pkl','wb') as out:
            pkl.dump({'distmat':distmat,'gqls':qls2},out)
        # break
    return 
