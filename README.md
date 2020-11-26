## Python implementation of K2 algorithm (Cooper-Herskovts, 1992). + K3 metric - Structural Learning of Bayesian Networks

This repository contains a script containing the python implementation of the K2 algorithm, used for structural learning of Bayesian Networks, using two score metrics 

	* Cooper-Herskovits. (original score metric)
	* Minimum description Length. (Known as K3, proposed by Bouckaert, 1993)
	
By default the algorithms learns the Network structure using Cooper-Herskovits score metric, but you can easily uncomment some lines (specified in the code) in the implementation to use MDL score. 

If you want to use large data, it is possible to compute the Cooper-Herskovits using the sum of factorials, instead of just using factorials, to prevent the high computational cost. See the commented lines in *f_ch* function 

For study purposes, I also provide a Jupyter Notebook containing: 
	* The implementation itself. 
	* Extra function to plot the Bayesian Networks from a dictionary. 
	* An example of usage of the algorithm, using fake data. 

References used to implement this algorithm: 
	* K2 algorithm 
		* Cooper, G.F.; Herskovits, E. A Bayesian method for the induction of probabilistic networks from data.766Machine learning1992,9, 309–347 
	* MDL score metric 
		* Bouckaert, R.R. Probabilistic network construction using the minimum description length principle.763European conference on symbolic and quantitative approaches to reasoning and uncertainty. Springer,7641993, pp. 41–48
	


