import pprint

class Graph():

    def __init__(self, matrix, thresh = 5e-5): #assumes concentration matrix

        self.neighbours = {}
        self.edges = {}
        self.node_vals = {}
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                
                v = matrix[i,j]

                if v < -thresh or v > thresh:
                    
                    if i not in self.neighbours:
                        self.neighbours[i] = set()

                    self.neighbours[i].add( j )

                    if j not in self.neighbours:
                        self.neighbours[j] = set()

                    self.neighbours[j].add( i )

                    self.edges[(i,j)] = v
                    self.edges[(j,i)] = v

    def get_neigh(self, idx):
        return self.neighbours[idx]

    def get_edge(self, idxa, idxb):
        return self.edges[idxa, idxb]

    def info(self):

        pp = pprint.PrettyPrinter(indent=4)
        
        print("node neighbours: ")
        pp.pprint(self.neighbours)
        print("edges: ")
        pp.pprint(self.edges)

    def set_node_vals(self, idx, val):
        self.node_vals[idx] =  val

    def get_node_vals(self, idx):
        return self.node_vals[idx]
        
    def optimize(self):
        
        pass
    
        # todo: implement belief propagation
        # eg: see section 2.3 of Gaussian Belief Propagation: Theory and Application by Danny Bickson
       
        # stop = False
        
        # while not stop:
            
            
