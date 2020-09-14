import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X = None, ADV = None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X = X, ADV = ADV)
        # Generate reports once in the beginning and save them, since
        # we do not have access to the distributions
        self.reports = X[0] # TODO: set X also in build_generator for other modes than train

        # Note on shapes:
        # X shape: [batch_size, num_agents, num_items]
        # X[0]: [self.num_agents, self.num_items]
        # ADV shape: [num_misreports, batch_size, num_agents, num_items]
    
    #TODO: Do online, to drop perm?

    # Note: Should not be used if X is provided
    # Repeat reports in the correct shape
    def generate_random_X(self, shape):
        a, _, _ = shape
        X = np.stack(((self.reports,)*a))
        #print(X)
        return X

    def generate_random_ADV(self, shape):
        # Misreport computation during auction learning does not
        # change. Regret is used to learn one auction per set of reports.
        # We don't need to *shape with random.binomial
        return(np.random.binomial(1, 0.5, shape))
