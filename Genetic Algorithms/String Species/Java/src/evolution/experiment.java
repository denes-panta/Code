package evolution;

import java.util.*;

public class Experiment{
	
	public int population; // population size
	public String prototype; // the string the program is looking for
	public int protoScore; // length of the prototype / max score
	public String[] speciesArray; // Array for the population
	public double[] scoreArray; // Array for the scores
	public int generation = 0; // genearion counter
	public int nParents; // number of parents
	public double mutationProb; // mutation probability
	
	
	public String[] generatePopulation(int pop_n, int prototype_score) {
		// Method for generating population\
		// pop_n <- population size
		// prorotype score <- protoScore
		String[] popArray = new String[pop_n]; // local array to store scores
		Arrays.fill(popArray, ""); // fill the array with "" to eliminate NaNs
		
		for (int i=0; i<pop_n; i++) { // generate random strings
			for (int j=0; j< prototype_score; j++) { // length of the prototype
				popArray[i] += (char) (Math.floor(Math.random() * (126 - 32)) + 32); // random character from ASCII table
			}
		}
		return popArray;
	}
	
	
	public double formula(double sc) {
		// Method for calculating the score for each speciment
		// sc <- input fittness score
		sc = Math.pow(sc, 2);
		return sc;
	}
	
	
	public String paragon(int pop_n, double[] fittnessScores, String[] stringSpecies) {
		// Method for getting the best speciment from the population
		// pop_n <- population size
		// fittnessScores <- Array containing the fittness scores
		// stringSpecies <- Array containing the speciment of the species
		int paragonIndex = 0; //index of the paragon, set it to the 0th item.
		
		for (int i=1; i<pop_n; i++){ // iterate over the population from Item with the 1st index
			if (fittnessScores[i] > fittnessScores[paragonIndex]) { // if the i-th indexed item has higher score than the current paragon item,
				paragonIndex = i; // then set paragonIndex to the i-th index
			}			
		}
		return stringSpecies[paragonIndex]; // return the paragon item
	}
	
	
	public double[] calculateScores(int pop_n, int prototype_score, String stringPrototype, String[] stringSpecies) {
		// Method for calculating the fitness scores
		// pop_n <- population size
		// prototype_score <- Maximum score / total length of the prototype
		// stringSpecies <- Array containing the speciment of the species
		double[] scArray = new double[pop_n]; // local Array for fittness scores
		
		for (int i=0; i<pop_n; i++) { //for all he species in the population
			for (int j=0; j<prototype_score; j++) { // for all the characters (cells) in the species
				if (stringSpecies[i].toCharArray()[j] == stringPrototype.toCharArray()[j]) {
					scArray[i] += 1; // count how many characters match the prototype
				}
			}
			scArray[i] = formula(scArray[i]); // calculate the fittness score based on a formula
		}		
		
		double sumArray = Arrays.stream(scArray).sum(); // calculate the sum for the score array
		for (int i=0; i<pop_n; i++) { // calculate relative probabilities for each species
			scArray[i] = scArray[i] / sumArray;
		}
		return scArray;
	}
	
	
	public String procreate(String evolType, int k, double mutaProb, int prototype_score, int n_pop, double[] fittnessScores, String[] stringSpecies) {
		// Method for procreating a new generation of string species
		// evolType <- Type of evolution
		// k <- number of parents
		// mutaProb <- Probability of random mutation during the reproduction
		// prototype_score <- Maximum score / total length of the prototype
		// pop_n <- population size
		// fittnessScores <- Array of scores 1:pop_n
		// stringSpecies <- Array containing the speciment of the species
		nParents = k;
		mutationProb = mutaProb;
		ArrayList<String> contestants = new ArrayList<String>(); // ArrayList for the possible parents
		int s = (int) Math.ceil(prototype_score/2); // split point for the speciments
		String child = ""; //create an initial String child speciment
		
		if (k != 2) { // if more then 2 parent is used for procreation, automatically user random evolution
			evolType = "rand"; // set evolution type to rando
		}
		
		do { // select a pool of candidates for parents
			for (int i=0; i<n_pop; i++) { // iterate over the entire population for fairness
				if (fittnessScores[i] > Math.random() * (1 - 0)){
					contestants.add(stringSpecies[i]);  // add to the pool of contestants based on probability
				}
			}
		} while(contestants.size() < k);

		if (evolType == "rand") { // random evolution
			for (int i=0; i<prototype_score; i++) {	
				int mask = (int) (Math.floor(Math.random() * (k - 0))); // randomly select a parent from the contestants pool
				
				if ((Math.random() * (1 - 0)) <= mutaProb) { 
					child += (char) (Math.floor(Math.random() * (126 - 32)) + 32); // mutate the cells based on mutaProb variables
				} else {
					child += contestants.get(mask).charAt(i); // take the cells from a randomly selected parent
				}
			}
		} else if (evolType == "split") { // split evolution
			child += contestants.get(0).substring(0, s); // take the first half of the 1st parent
			child += contestants.get(1).substring(s, prototype_score); // take the second half of the 2nd parent

			for (int i=0; i<prototype_score; i++) { // mutate the cells based on mutaProb variables
				if ((Math.random() * (1 - 0)) <= mutaProb) {
					child.toCharArray()[i] = (char) (Math.floor(Math.random() * (126 - 32)) + 32);
				}
			}
		}
		
		return child;
	}
	
	
	public Experiment(int createPopulation, String definePrototype) {
	    this.population = createPopulation; // set population
	    this.prototype = definePrototype; // set prototype
	    this.protoScore = definePrototype.length(); // set protoScore
	    this.speciesArray = generatePopulation(this.population, this.protoScore); // generate initial population
	    this.scoreArray = calculateScores(this.population, this.protoScore, this.prototype, this.speciesArray); // calculate initial scores
	}
	
	
	public static void main(String[] args) {
		Experiment stringExp = new Experiment(500, "The Orville is a great TV show!"); // create new experiment and pass population and prototype
		
		while (stringExp.paragon(stringExp.population, stringExp.scoreArray, stringExp.speciesArray).equals(stringExp.prototype) == false) { //do until the paragon of the species matches the prototype
			stringExp.generation += 1; // increment generation counter

			for (int species=0; species<stringExp.population; species++) { // create new generation
				stringExp.speciesArray[species] = stringExp.procreate("rand", 2, 0.01, stringExp.protoScore, stringExp.population, stringExp.scoreArray, stringExp.speciesArray); // set evolution type, number of parents, mutaProb
			}
			stringExp.scoreArray = stringExp.calculateScores(stringExp.population, stringExp.protoScore, stringExp.prototype, stringExp.speciesArray); // calculate scores
			if (stringExp.generation % 10 == 0) {
				System.out.println(stringExp.generation + ". :" + stringExp.paragon(stringExp.population, stringExp.scoreArray, stringExp.speciesArray));	// Print paragon and generion number every 10 iteration	
			}
		}
		// Print final assessment of the evolution
		System.out.println("---------------------------");
		System.out.println("Final generaion: " + stringExp.generation);
		System.out.println( "Paragon :"+ " " + stringExp.paragon(stringExp.population, stringExp.scoreArray, stringExp.speciesArray));
		System.out.println("Population: " + stringExp.population);
		System.out.println("Speciment size: " + stringExp.protoScore);
		System.out.println("Number of Parents: " + stringExp.nParents);
		System.out.println("Mutation Probability: " + (int) (stringExp.mutationProb * 100) + "%");
	}
}
