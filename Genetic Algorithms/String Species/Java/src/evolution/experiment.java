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
	
	public String[] generatePopulation() {
		// Method for generating population
		String[] popArray = new String[population]; // local array to store scores
		Arrays.fill(popArray, ""); // fill the array with "" to eliminate NaNs
		
		for (int i=0; i<population; i++) { // generate random strings
			for (int j=0; j< protoScore; j++) { // length of the prototype
				popArray[i] += (char) (Math.floor(Math.random() * (126 - 32)) + 32); // random character from ASCII table
			}
		}
		return popArray;
	}
	
	
	public double formula(double sc) {
		//Method for calculating the score for each speciment
		sc = Math.pow(sc, 2);
		return sc;
	}
	
	
	public String paragon() {
		// Method for getting the best speciment from the population
		int paragonIndex = 0; //index of the paragon, set it to the 0th item.
		
		for (int i=1; i<population; i++){ // iterate over the population from Item with the 1st index
			if (scoreArray[i] > scoreArray[paragonIndex]) { // if the i-th indexed item has higher score than the current paragon item,
				paragonIndex = i; // then set paragonIndex to the i-th index
			}			
		}
		return speciesArray[paragonIndex]; // return the paragon item
	}
	
	
	public double[] calculateScores() {
		// Method for calculating the fitness scores
		double[] scArray = new double[population]; // local Array for fittness scores
		
		for (int i=0; i<population; i++) { //for all he species in the population
			for (int j=0; j<protoScore; j++) { // for all the characters (cells) in the species
				if (speciesArray[i].toCharArray()[j] == prototype.toCharArray()[j]) {
					scArray[i] += 1; // count how many characters match the prototype
				}
			}
			scArray[i] = formula(scArray[i]); // calculate the fittness score based on a formula
		}		
		
		double sumArray = Arrays.stream(scArray).sum(); // calculate the sum for the score array
		for (int i=0; i<population; i++) { // calculate relative probabilities for each species
			scArray[i] = scArray[i] / sumArray;
		}
		return scArray;
	}
	
	
	public String procreate(String evolType, int k, double mutaProb) {
		//MEthod for procreating a new generation of string species
		nParents = k;
		mutationProb = mutaProb;
		ArrayList<String> contestants = new ArrayList<String>(); // ArrayList for the possible parents
		int s = (int) Math.ceil(protoScore/2); // split point for the speciments
		String child = ""; //create an initial String child speciment
		
		if (k != 2) { // if more then 2 parent is used for procreation, automatically user random evolution
			evolType = "rand"; // set evolution type to rando
		}
		
		do { // select a pool of candidates for parents
			for (int i=0; i<population; i++) { // iterate over the entire population for fairness
				if (scoreArray[i] > Math.random() * (1 - 0)){
					contestants.add(speciesArray[i]);  // add to the pool of contestants based on probability
				}
			}
		} while(contestants.size() < k);

		if (evolType == "rand") { // random evolution
			for (int i=0; i<protoScore; i++) {	
				int mask = (int) (Math.floor(Math.random() * (k - 0))); // randomly select a parent from the contestants pool
				
				if ((Math.random() * (1 - 0)) <= mutaProb) { 
					child += (char) (Math.floor(Math.random() * (126 - 32)) + 32); // mutate the cells based on mutaProb variables
				} else {
					child += contestants.get(mask).charAt(i); // take the cells from a randomly selected parent
				}
			}
		} else if (evolType == "split") { // split evolution
			child += contestants.get(0).substring(0, s); // take the first half of the 1st parent
			child += contestants.get(1).substring(s, protoScore); // take the second half of the 2nd parent

			for (int i=0; i<protoScore; i++) { // mutate the cells based on mutaProb variables
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
	    this.speciesArray = generatePopulation(); // generate initial population
	    this.scoreArray = calculateScores(); // calculate initial scores
	}
	
	
	public static void main(String[] args) {
		Experiment stringExp = new Experiment(1000, "The Orville is a great TV show!"); // create new experiment and pass population and prototype
		
		while (stringExp.paragon().equals(stringExp.prototype) == false) { //do until the paragon of the species matches the prototype
			stringExp.generation += 1; // increment generation counter

			for (int species=0; species<stringExp.population; species++) { // create new generation
				stringExp.speciesArray[species] = stringExp.procreate("rand", 2, 0.01); // set evolution type, number of parents, mutaProb
			}
			stringExp.scoreArray = stringExp.calculateScores(); // calculate scores
			if (stringExp.generation % 10 == 0) {
				System.out.println(stringExp.generation + ". :" + stringExp.paragon());	// Print paragon and generion number every 10 iteration	
			}
		}
		// Print final assessment of the evolution
		System.out.println("---------------------------");
		System.out.println("Final generaion: " + stringExp.generation);
		System.out.println( "Paragon :"+ " " + stringExp.paragon());
		System.out.println("Population: " + stringExp.population);
		System.out.println("Speciment size: " + stringExp.protoScore);
		System.out.println("Number of Parents: " + stringExp.nParents);
		System.out.println("Mutation Probability: " + (int) (stringExp.mutationProb * 100) + "%");
	}
}
