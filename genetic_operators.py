# In this file we save the operators for the evolutionary algorithm

#Import libraries
import numpy as np


class GeneticOperators:
    def __init__(self, env, dom_u, dom_l, npop, gens, mutation, n_vars):
        self.env = env
        self.dom_u = dom_u
        self.dom_l = dom_l
        self.npop = npop
        self.gens = gens
        self.mutation = mutation
        self.n_vars = n_vars

    def simulation(self, x):
        f,p,e,t = self.env.play(pcont=x)
        return f


    #main operators
    def evaluate(self, x):
        return np.array(list(map(lambda y: self.simulation(y), x)))
    
    def parent_selection(self, fit_pop, pop):
        c1 =  np.random.randint(0,pop.shape[0], 1)
        c2 =  np.random.randint(0,pop.shape[0], 1)

        if fit_pop[c1] > fit_pop[c2]:
            return pop[c1][0]
        else:
            return pop[c2][0]

    
    def crossover(self, fit_pop, pop):
        total_offspring = np.zeros((0,self.n_vars))


        for p in range(0,pop.shape[0], 2):
            p1 = self.parent_selection(fit_pop, pop)
            p2 = self.parent_selection(fit_pop, pop)

            n_offspring =   np.random.randint(1,3+1, 1)[0]
            offspring =  np.zeros( (n_offspring, self.n_vars) )

            for f in range(0,n_offspring):

                cross_prop = np.random.uniform(0,1)
                offspring[f] = p1*cross_prop+p2*(1-cross_prop)

                # mutation
                for i in range(0,len(offspring[f])):
                    if np.random.uniform(0 ,1)<=self.mutation:
                        offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

                offspring[f] = np.array(list(map(lambda y: self.limits(y), offspring[f])))

                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring
    
    def survivor_selection(self, pop, fit_pop, best):
        fit_pop_cp = fit_pop
        fit_pop_norm =  np.array(list(map(lambda y: self.norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
        probs = (fit_pop_norm)/(fit_pop_norm).sum()
        chosen = np.random.choice(pop.shape[0], self.npop , p=probs, replace=False)
        chosen = np.append(chosen[1:],best)
        pop = pop[chosen]
        fit_pop = fit_pop[chosen]
            
        return pop, fit_pop
    
    def doomsday(self, pop, fit_pop):

        worst = int(self.npop/4)  # a quarter of the population
        order = np.argsort(fit_pop)
        orderasc = order[0:worst]

        for o in orderasc:
            for j in range(0,self.n_vars):
                pro = np.random.uniform(0,1)
                if np.random.uniform(0,1)  <= pro:
                    pop[o][j] = np.random.uniform(self.dom_l, self.dom_u) # random dna, uniform dist.
                else:
                    pop[o][j] = pop[order[-1:]][0][j] # dna from best

            fit_pop[o]=self.evaluate([pop[o]])

        return pop,fit_pop
            

    #auxilary operators
    def limits(self, x):
        if x>self.dom_u:
            return self.dom_u
        elif x<self.dom_l:
            return self.dom_l
        else:
            return x
        
    def norm(self, x, pfit_pop):
        if ( max(pfit_pop) - min(pfit_pop) ) > 0:
            x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
        else:
            x_norm = 0

        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm

