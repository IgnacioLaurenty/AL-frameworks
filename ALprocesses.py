import os
import pickle
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from ALDataset import ALDataset
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances
#from mip import Model, xsum, BINARY, CBC, OptimizationStatus  #import needed only for Coreset_Robust
from batchbald_redux_subpack import get_batchbald_batch
from typing import Optional, List, Tuple, Union, TypeVar
T = TypeVar('T')


class ALModule() :
    """
    Generic class for Active Learning processes, which can basically perform Random Sampling.
    This class needs the methods define_model, train, test to be overridden.
    If you don't plan to save the statistics, please overridde also the save method.

    Parameters
    ----------
    train_data : ALDataset
        Dataset on with Active Learning is performed.

    test_data : Dataset
        Dataset for evaluation of the Active Learning process

    Arguments 
    ----------
    init_lab_prop : float in [0,1[, default=0.05
        proportion of train_data labeled and used as initialization of an Active Learning process
    
    init_lab_size : int, default=None
        size of the initial training set. Override init_lab_prop if is int.

    anno_prop : float, default=0.1
        gives the annotation batch size relatively to the train_data length.
    
    num_pts : int, default=None
        If int, override anno_prop to give the annotation batch size.

    saving_dir : str, default=None
        root in which the statistics of the Active Learning process are saved
    
    Attributes
    ----------
    stats : dict
        Dictionnary saving accuracy and loss on the test dataset ;
        saving annotation cost per round and cumulatively. 

    anno_batch_size : int
        number of new labels at each Active Learning step.
        Default : 10% of the train dataset.
    
    init_lab_size : int
        number of labeled points for initialization.
        Default : 5% of the train dataset.
    
    file_name : str, default=None
        to be defined if you want to save the statistics dictionnary.

    """
    def __init__(self, train_data : ALDataset, test_data : Dataset, **kwargs):
        self.train_data = train_data
        self.test_data = test_data        
        self.file_name = None
        self.saving_dir = kwargs.get('saving_dir', None)
        self.anno_prop = kwargs.get('anno_prop', 0.1)
        self.anno_batch_size = kwargs.get('num_pts', int(len(self.train_data.remaining_indices())*self.anno_prop) )
        self.init_lab_prop = kwargs.get('init_lab_prop', 0.05)
        self.init_lab_size = kwargs.get('init_lab_size', int(len(self.train_data.remaining_indices())*self.init_lab_prop))
    
    def cost(self, idxs : List[T]) -> int :
        '''
        Default : return a cost equal to 1 for each new labeled sample
        '''
        return len(idxs)
    
    def initialize_process(self) -> None :
        init_idx = np.random.permutation(self.train_data.remaining_indices())
        init_idx = init_idx[:self.init_lab_size]
        self.train_data.true_labels(init_idx)

    def budget(self) -> bool :
        '''
        return True while the fixed budget is not reached.
        '''
        return len(self.train_data.remaining_indices())>0    

    def define_model(self):
        raise NotImplementedError("please implement the 'define_model' method")
    
    def train(self):
        raise NotImplementedError("please implement the 'train' method")
    
    def test(self):
        '''
        Should return accuracy and loss on the test set
        '''
        raise NotImplementedError("please implement the 'test' method")


    def get_batch_examples(self) -> np.ndarray :
        '''
        this default implementation is the acquisition function of Random Sampling
        '''
        return np.random.permutation(self.train_data.remaining_indices())[:self.anno_batch_size]

         
    def label(self, idx : Union[int, List[int]]) -> None :
        '''
        The method can be modified to update the labels with partial labels using the update_labels method from ALDataset
        '''
        self.train_data.true_labels(idx)

    def update_stats(self, cost : float ) -> None :
        test_acc, test_loss = self.test()
        try:
            self.stats['acc'].append(test_acc)
            self.stats['loss'].append(test_loss)
            self.stats['round_cost'].append(cost)
            self.stats['cumul_cost'].append(self.stats['cumul_cost'][-1]+cost)
        except :
            self.stats = {'acc' : [test_acc],
                          'loss' : [test_loss],
                          'round_cost' :[cost],
                          'cumul_cost' :[cost] }

    def set_file_name(self, file_name : str) -> None :
        self.file_name = file_name

    def save(self, file_name : str ) -> None :
        assert type(self.saving_dir) == str
        with open(os.path.join(self.saving_dir,file_name),'wb') as f:
            pickle.dump(self.stats,f)


    def work(self) -> None :
        assert type(self.file_name) ==str, "please set a file_name attribute before using the work method"
        print('initializing AL process')
        self.initialize_process()
        self.define_model()
        if len(self.train_data.train_indices()) > 0:
            self.train()
        self.update_stats(cost = self.cost(self.train_data.train_indices()))
        count =1
        while self.budget():
            print('Active Learning step n°{}'.format(count))
            idx_to_label = self.get_batch_examples()
            self.label(idx_to_label)
            print('training substep')
            self.define_model() 
            self.train()
            self.update_stats( cost = self.cost(idx_to_label))
            self.save(self.file_name)
            count += 1


class Uncertainty_Sampling(ALModule):
    """
    Generic class for Uncertainty sampling based on ALModule class.
    This class can be used by overridding the following methods :
        train
        test
        define_model
        get_bayesian_probs : must return an array-like of shape (K, num_unlabeled, num_classes), 
                             where K >= 1

    Original papers
    ----------
    BALD : 
        Deep Bayesian Active Learning with Image Data
        Yarin Gal and Riashat Islam and Zoubin Ghahramani
        Proceedings of the 34th International Conference on Machine Learning, 2017
        https://proceedings.mlr.press/v70/gal17a.html
                             
    Arguments
    ----------
    mode : str, default='entropy'
        available modes are : 'entropy', 'bald', 'least confident', 'margin', 'random'

    see ALModule for other parameters, arguments and attributes.
    """
    def __init__(self,  train_data : ALDataset, test_data : Dataset, mode : str = 'entropy', **kwargs) -> None :
        super().__init__( train_data= train_data, test_data = test_data, **kwargs)
        self.mode = mode
    
    def get_bayesian_probs(self):
        raise NotImplementedError('please implement this method to output an array-like of shape (average_size, num_unlabeled, num_classes)')

    def get_batch_examples(self) -> np.ndarray:
        if self.mode == 'bald' :
            probs = self.get_bayesian_probs()
            H = Categorical(probs.mean(0)).entropy()
            E_H = - (torch.sum(probs * torch.log(probs + 1e-10), axis = -1)).mean(0)
            idx = torch.argsort(H-E_H)[-self.anno_batch_size:].numpy()
            return self.train_data.remaining_indices()[idx]

        elif self.mode in ['entropy','least confident', 'margin' ] :
            def calc(probs):
                if self.mode == 'entropy' :
                    return Categorical(probs=probs).entropy()
                elif self.mode == 'least confident' :
                    return 1 - probs.max(-1).values
                elif self.mode == 'margin' :
                    sorted_probs= probs.sort(-1).values
                    return sorted_probs[:,-1]-sorted_probs[:,-2]
                    
            probs = self.get_bayesian_probs()
            probs = probs.mean(0)
            measures = calc(probs)
            if self.mode =='margin' :
                idx = torch.argsort(measures)[:self.anno_batch_size].numpy()
            else:
                idx = torch.argsort(measures)[-self.anno_batch_size:].numpy()
            return self.train_data.remaining_indices()[idx]
            
        elif self.mode == 'random' :
            return np.random.permutation(self.train_data.remaining_indices())[:self.anno_batch_size]
        else :
            raise NotImplementedError('{} not implemented'.format(self.mode))


class Coreset_Greedy(ALModule):
    """
    Generic class implementing the Coreset greedy framework.
    This class can be used by overridding the following methods :
        train
        test
        define_model
        get_features : must return an array-like (train_data.__len__(), feature_dim) embedding the train dataset. 
    
    Original paper
    ----------
    Active Learning for Convolutional Neural Networks: A Core-Set Approach, Ozan Sener and Silvio Savarese
    International Conference on Learning Representations, 2018
    https://arxiv.org/abs/1708.00489
    
    Arguments :
    ----------
    metric : str, default='euclidean'
        see sklearn.metrics.pairwise_distances for details.

    see ALModule for other parameters, arguments and attributes.
    """
    def __init__(self, train_data : ALDataset, test_data : Dataset, metric : str ='euclidean', **kwargs) -> None :
        super().__init__( train_data = train_data, test_data = test_data, **kwargs)
        self.metric = metric

    def compute_distance(self, X : Union[np.ndarray, Tensor], x : Optional[Union[np.ndarray, Tensor]]) -> Union[np.ndarray, Tensor] :
        return pairwise_distances(X, x , metric=self.metric)

    def get_features(self):
        raise NotImplementedError('please implement the method to return the representation over the dataset')

    def update_distances(self, cluster_centers, reset_dist : bool =False, pool : Optional[List[int]] = None) -> None :
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          rest_dist: whether to reset min_distances.
        """
        if pool is None:
            pool = self.train_data.remaining_indices()

        if reset_dist:
            self.features = self.get_features()

        x = self.features[cluster_centers]
        # Update min_distances for all examples given new cluster center.
        dist = self.compute_distance(self.features[pool], x) 

        if reset_dist:
            self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def get_batch_examples(self, pool = None, return_delta : bool = False, train_subset = None) ->Union[Tuple[List[int], float], List[int]] :
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        centers are the labeled datapoints plus the currently selected datapoints for annotation.
        Returns:
          indices of points selected to maximize distance to cluster centers
        """
        if pool is None :
            pool = self.train_data.remaining_indices()
        if train_subset is None :
            train_subset = self.train_data.classified_indices()

        self.update_distances(cluster_centers = train_subset, reset_dist=True, pool = pool)
        new_batch = []
        if len(self.train_data.remaining_indices()) > self.anno_batch_size:
            for _ in range(self.anno_batch_size):
                if len(self.train_data.classified_indices()) == 0 :
                    new_batch = np.random.choice(self.train_data.remaining_indices(), self.anno_batch_size, replace = False).tolist()
                    break
                else:
                    ind = np.argmax(self.min_distances)
                    ind = pool[ind]
                assert ind not in self.train_data.classified_indices()
                assert ind in pool
                assert ind not in new_batch  
                #pool = np.delete(pool, np.where(pool == ind))
                self.update_distances([ind], reset_dist=False, pool = pool)
                new_batch.append(ind)
        else :
            new_batch=self.train_data.remaining_indices().tolist()

        if return_delta:
            self.update_distances(new_batch, reset_dist=True, pool = pool)
            delta = self.min_distances.max()
            return delta, new_batch
        else :
            return new_batch  



class BatchBALD(ALModule):
    """
    Generic class for BatchBALD algorithm.
    This class can be used by overridding the following methods :
        train
        test
        define_model
        get_bayesian_probs : must return an array-like of shape (num_unlabeled, K, num_classes), 
                             where K >= 1, with first dimension sorted like self.train_data.remaining_indices()
                             
    Original code : https://github.com/blackhc/batchbald_redux, Apache-2.0 Licence, Package name : batchbald_redux

    Original paper
    ----------
    BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning
    Andreas Kirsch, Joost van Amersfoort, Yarin Gal, 2019
    https://arxiv.org/abs/1906.08158

    Attributes
    ----------
    K : int, default=100
        number of iterations to estimate the expectations

    restrict : int, default=None
        if int, restrict the batch construction on random subsets of the unlabeled set of size restrict

    see ALModule for other parameters, arguments and attributes.
    """
    def __init__(self, train_data : ALDataset, test_data : Dataset, K : int = 100 , **kwargs) -> None :
        super().__init__(train_data, test_data, **kwargs)
        self.K = K
        self.restrict = kwargs.get('restrict', None)
        if self.restrict is not None:
            assert type(self.n_restrict) == int and self.restrict > self.anno_batch_size, "restrict must be an integer larger than annotation batch size"


    def get_bayesian_probs(self):
        raise NotImplementedError("please implement to return a Tensor of probabilities with shape (len(unlabeled_set), self.K, num_classes)")
    
    def get_batch_examples(self) -> np.ndarray:
        prob_NKC = self.get_bayesian_probs()
        if self.restrict is not None and self.restrict < len(self.train_data.remaining_indices()) :
            pool = np.random.choice(np.arange(len(self.train_data.remaining_indices())), self.restrict, replace = False )
        else : 
            pool = np.arange(len(self.train_data.remaining_indices()))
        with torch.no_grad():
            batch = get_batchbald_batch(prob_NKC[pool], self.anno_batch_size, len(pool), device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') )
        return self.train_data.remaining_indices()[pool[batch.indices]]
    
    
    



class ALPF(ALModule):
    """
    Generic class implementing the ALPF algorithm.
    This class can be used by overridding the following methods :
        train
        test
        define_model
        get_probs : must return an array-like of probabilies of shape (self.train_data.remaining_indices(), num_classes),
                    ordered as self.train_data.remaining_indices()
    
    Original paper
    ----------
    Active Learning with Partial Feedback,
    Peiyun Hu, Zachary C. Lipton, Anima Anandkumar, Deva Ramanan,
    ICLR 2019
    https://arxiv.org/abs/1802.07427

    Parameters
    ----------
    Q : torch.Tensor, shape : (num_questions, num_classes)
        Tensor encoding the classes giving the binary questions to be answered.

    l_cost : float, default=0.
        can be used to modify the acquisition function as a linear trade-off between measure and question costs.

    Attributes
    ----------
    mode : str, default='eig'
        can be 'eig', 'edc', or 'erc'. see original paper for details.

    stats : dict
        recover statistics :
            test accuracy, test loss, annotation cost per round, cumulative annotation cost, questions indexes

    see ALModule for other parameters, arguments and attributes.
    """
    def __init__(self, train_data : ALDataset, test_data : Dataset, Q : Tensor, l_cost :float = 0., mode : str = 'eig', **kwargs) -> None:
        super().__init__(train_data = train_data, test_data = test_data, **kwargs)
        self.Q = Q.float()
        self.l_cost = l_cost
        self.mode = mode
        self.remaining_questions = torch.ones( (self.train_data.__len__(), len(self.Q)) )
        self.remaining_questions[ self.train_data.val_idx ] = 0

    def get_probs(self) :
        '''
        return a probability matrix of shape (n_remaining_partial_label, num_classes) ordered as self.train_data.remaining_indices()
        '''
        raise NotImplementedError('must be implemented by the user.')

    def initialize_process(self) -> None:
        super().initialize_process()
        self.remaining_questions[self.train_data.classified_indices()] = 0
    

    def expected_information_gain(self, prob_mat : Tensor) -> Tensor:
        qp = 1e-12+ prob_mat @ self.Q.transpose(0,1)
        mere = torch.nan_to_num(-(qp*torch.log(qp)+(1-qp)*torch.log(1-qp))) 
        return mere
    
    def cost(self, idx : Optional[Union[int, np.ndarray, Tensor]] = None) -> Union[np.ndarray, Tensor, float] :
        costs = np.ones(len(self.Q))
        if idx is None:
            return costs
        else :
            return costs[idx]
    
    def expected_decrease_classes(self, prob_mat : Tensor, y_partial : Tensor) -> Tensor:
        return (y_partial.sum(-1) - self.expected_remaining_classes(prob_mat,y_partial).t()).t()

    
    def expected_remaining_classes(self, prob_mat : Tensor, y_partial : Tensor) -> Tensor:
        qp = 1e-12+ prob_mat @ self.Q.transpose(0,1)
        y_1 = y_partial @ self.Q.transpose(0,1)
        y_0 = (y_1.max(-1)- y_1.t()).t()              
        return qp*y_1 + (1-qp)*y_0


    def update_remaining_questions(self, index : int) -> None :
        a = self.train_data.partial_labels[index] @ self.Q.transpose(0,1) >0
        b = self.train_data.partial_labels[index] @ self.Q.transpose(0,1) < self.train_data.partial_labels[index].sum(-1)
        self.remaining_questions[index] = a*b
        if self.remaining_questions[index].sum(-1)>0 : self.remaining_questions[index, self.Q.sum(-1)==self.train_data.num_classes] = 1.

    def label(self, idx : int, question : int) -> None :
        new_label = self.train_data.partial_labels[idx] * self.Q[question]
        if new_label[self.train_data.dataset.targets[idx]] == 0.: 
            new_label = self.train_data.partial_labels[idx] - new_label
        self.train_data.update_label(idx, new_label)
        self.update_remaining_questions(idx)


    def get_batch_examples(self, prob_mat : Tensor ) -> None :
        if self.mode =='eig':
            acquisition = self.expected_information_gain(prob_mat) - self.l_cost * self.cost()
        elif self.mode == 'edc' :
            acquisition = self.expected_decrease_classes(prob_mat, self.train_data.partial_labels[self.train_data.remaining_indices()] ) - self.l_cost * self.cost()
        elif self.mode =='erc' :
            acquisition = -(self.expected_remaining_classes(prob_mat, self.train_data.partial_labels[self.train_data.remaining_indices()] ) + self.l_cost*self.cost())
        else :
            raise NotImplementedError('mode {} not implemented'.format(self.mode))
        
        acquisition[torch.where(self.remaining_questions[self.train_data.remaining_indices()] == 0 )] = - torch.inf
        count = 0
        save_idx_round = self.train_data.remaining_indices().copy()
        while count < self.anno_batch_size and len(self.train_data.remaining_indices()) > 0 and self.budget() :
            count += 1
            idx = acquisition.max(1).values.argmax()
            question = acquisition[idx].argmax().item()
            self.label(save_idx_round[idx], question)
            acquisition[ idx, self.remaining_questions[save_idx_round[idx]] == 0 ] = - torch.inf
            self.stats['cumul_cost'][-1] += self.cost(question)
            self.stats['round_cost'][-1] += self.cost(question)
            self.stats['questions'][-1].append(question)

    def update_stats(self, cost : float) -> None :
        super().update_stats(cost)
        try :
            self.stats['questions'].append([])
        except :
            self.stats['questions'] = [[]]

    def work(self) -> None :
        assert self.file_name is not None, "please set a file_name attribute before using the work method"
        print('initializing AL process')
        self.initialize_process()
        self.define_model()
        if len(self.train_data.train_indices()) > 0:
            self.train()
        self.update_stats( cost = len(self.train_data.train_indices())*self.cost().max() )
        count =1
        while self.budget():
            print('Active Learning step n°{}'.format(count))
            prob_mat = self.get_probs()
            self.get_batch_examples(prob_mat)
            print('training substep')
            self.define_model() 
            self.train()
            self.update_stats( cost = 0)
            self.save(self.file_name)
            count += 1



class Coreset_Robust(Coreset_Greedy):      
    '''
    def __init__(self,  train_data : ALDataset, test_data, 
                 metric = 'euclidean', outlier_proportion : float = 0.001, **kwargs ):
        super().__init__( train_data= train_data, test_data = test_data, metric = metric, **kwargs)
        self.metric = metric
        self.outlier_proportion = outlier_proportion
        self.restrict_trainset = kwargs.get('restrict_trainset', False)
        self.trainset_restriction = kwargs.get('trainset_restriction', 50 * self.train_data.num_classes )
        self.restrict_pool = kwargs.get('restrict_pool', False)
        self.num_candidates = kwargs.get('num_candidate', 10 * self.anno_batch_size )
        self.search_time_limit = kwargs.get('search_time_limit', np.inf)

        if not self.restrict_pool or not self.restrict_trainset or self.search_time_limit == np.inf :
           print('WARNING : You chose to not restrict the feasibility_check. The annotation batch construction might be VERY slow.')
           print('restriction on centers :', self.restrict_trainset)
           print('restriction on pool :', self.restrict_pool)
           print('restrictions on search time :' , self.search_time_limit != np.inf)

        raise NotImplementedError('too slow with mip package as solver.')

    def compute_distance(self, X, x):
        return pairwise_distances(X, x , metric=self.metric)
    

    def feasibility_check(self, b, train_subset, pool_subset, all_dist, delta, n_out):
      """Solving robust k-Center MIP in order to see if robust k-Center cost of delta is feasible
      #Arguments:
        b : batch size
        train_subset : indexes of a subset of labeled datapoints
        pool_subset : indexes if a subset of unlabeled datapoints
        all_dist: distances between all (i.e. both training and candidate) embeddings used train and pool subsets
        delta: k-Center cost for which feasibility is being determined
        n_out: total outlier tolerance for robust k-Center
      #Returns:
        faisibility, batch, status of MIP
      """
      n= len(train_subset) + len(pool_subset)
      model = Model(solver_name=CBC)
      u = [model.add_var('u({})'.format(j), var_type = BINARY) for j in range(n)]
      E = [[model.add_var('E({},{})'.format(i,j),var_type=BINARY) for j in range(n)] for i in range(n)]
      w = [[model.add_var('w({},{})'.format(i,j),var_type=BINARY) for j in range(n)] for i in range(n)]

      for i in range(n):
          model.add_constr( xsum(w[i][j] for j in range(n)) == 1, "cons{}".format(i+1) )
      for j in range(len(train_subset)) :
          model.add_constr(u[j] == 1, 'cons_s0')
      for i in range(n):
          for j in range(n):
              model.add_constr(w[i][j]<=u[j], 'cons_w{},{}<u{}'.format(i,j,j))
              if all_dist[i,j]>delta:
                  model.add_constr(w[i][j] == E[i][j], 'cons_wE({},{})'.format(i,j))

      model.add_constr( xsum( xsum(E[i][j] for j in range(n)) for i in range(n)) <= n_out , 'consE' )
      model.add_constr( xsum( u[i] for i in range(n) ) == len(train_subset) + b )
      model.objective = xsum(u[j] for j in range(n))
      status = model.optimize(max_seconds=self.search_time_limit)
      validity = status == OptimizationStatus.OPTIMAL
      batch = []
      for k in range(len(train_subset),n) :
         if u[k].x == 1. :
            batch.append(pool_subset[k-len(train_subset)])

      return validity, batch, status

    
    def farthest_first_kcenters(self, pool, train_subset):
      return super().get_batch_examples( pool = pool, return_delta = True, train_subset=train_subset)

    def get_batch_examples(self) :
      print('getting annotation batch')
      #Sampling pool candidates and train subset
      if self.restrict_pool :
         pool_ind_sample = np.random.choice(self.train_data.remaining_indices(), self.num_candidates, replace=False) if len(self.train_data.remaining_indices())>self.num_candidates else self.train_data.remaining_indices()
      else :
         pool_ind_sample = self.train_data.remaining_indices()
      
      if self.restrict_trainset:
        classified_idx_restricted = np.random.choice(self.train_data.classified_indices(), self.trainset_restriction, replace = False ) if len(self.train_data.classified_indices())>self.trainset_restriction else self.train_data.classified_indices()
      else : 
        classified_idx_restricted = self.train_data.classified_indices()

      batch_size = self.anno_batch_size
      
      #Initializing upper bound ub and lower bound ub for binary search on robust k-Center cost
      cost, batch = self.farthest_first_kcenters(pool = pool_ind_sample, train_subset = classified_idx_restricted )

      ub = cost
      lb = cost/2
      

      pairwise_distances = self.compute_distance(self.features[np.concatenate((classified_idx_restricted,np.array(pool_ind_sample)))], 
                                                 self.features[np.concatenate((classified_idx_restricted,np.array(pool_ind_sample)))])

      #Binary search on robust k-Center cost
      counter=1
      all_batches = [batch]
      while ub-lb >1e-7 :
        validity, batch, status = self.feasibility_check(b = batch_size, train_subset = classified_idx_restricted, pool_subset=pool_ind_sample,
                                                         all_dist = pairwise_distances, delta = (ub+lb)/2., n_out = self.outlier_proportion*self.num_candidates)
        all_batches.append(batch)
        counter +=1
        if validity :
          ub = np.amax(pairwise_distances[pairwise_distances <= (ub+lb)/2.0])
        else :
          lb = np.amin(pairwise_distances[pairwise_distances >= (ub+lb)/2.0])
      for ba in all_batches:
         if len(ba)>0:
            batch = ba
      return batch'''
    pass
    # raise NotImplementedError("not working without strong MIP solver. Please refer to the original code and paper for a good implementetation.")



