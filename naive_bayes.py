import numpy as np

class Naive_Bayes:
    
    def __init__(self, data):
        self.classes = np.unique(data[:,-1])
        self.features = np.arange(len(data[0, :-1])) 
        self.stats = self.get_stats(data)
        self.priors = self.get_priors(data)
        
        
    def get_priors(self, data):    
        priors = {}
        
        for c in self.classes:
            mask = data[:,-1] == c
            prior = np.sum(mask)/len(data)
            
            priors[c] = prior
        
        return priors
        
    def get_stats(self, data):
        classes = np.unique(data[:,-1])
        
        stats = {}
        
        for c in classes:
            feature_dict = {}
            
            mask = data[:,-1] == c
#             features = np.arange(len(data[0, :-1]))
            for feature in self.features:
                mean = np.mean(data[mask, feature])
                std = np.std(data[mask, feature])
                feature_dict[feature] = (mean, std)
            
            stats[c] = feature_dict
        
        return stats
    
        
    def norm_pdf(self, x, mu, sigma):
        first = (1/(sigma*np.sqrt(2*np.pi)))
        second = np.exp(-((x-mu)**2)/(2*(sigma**2)))
        return first*second
        
    
    def predict(self, x):
        
        class_probs = []
        for c in self.classes:
#             print(f'Class {c}')
            feature_probs = []
            for feature in self.features:
#                 print(f'Feature {feature}')
                mean = self.stats[c][feature][0]
#                 print(f'Mean: {mean}')
                std = self.stats[c][feature][1]
#                 print(f'Std: {std}')
#                 print(f'x value: {x[feature]}')
                if std > .0001:
                    prob = self.norm_pdf(x[feature], mean, std)
                    feature_probs.append(prob)
            class_prob = self.priors[c]*np.prod(feature_probs)
            class_probs.append(class_prob)
            
            
        naive_class_probs = []
        for c in class_probs:
            naive_class_prob = c/np.sum(class_probs)
            naive_class_probs.append(naive_class_prob)
            
#         print(naive_class_probs)
        return self.classes[np.argmax(naive_class_probs)]