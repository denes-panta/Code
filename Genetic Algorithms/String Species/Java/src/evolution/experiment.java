package evolution;

import java.util.*;

public class experiment{
	
	public int population;
	public String prototype;
	public int protoScore;
	public String[] speciesArray;
	public double[] scoreArray;
	public int generation = 0;
	
	public String[] generatePopulation() {
		String[] popArray = new String[population];
		Arrays.fill(popArray, "");
		
		for (int i=0; i<population; i++) {
			for (int j=0; j< protoScore; j++) {
				popArray[i] += (char) (Math.floor(Math.random() * (126 - 32)) + 32);
			}
		}
		return popArray;
	}
	
	
	public double formula(double score) {
		score = Math.pow(score, 2);
		return score;
	}
	
	
	public String paragon() {
		int maxIndex = 0; 
		for (int i=1; i<population; i++){
			if (scoreArray[i] > scoreArray[i-1]) {
				maxIndex = i;
			};			
		}
		return speciesArray[maxIndex];
	}
	
	
	public double[] calculateScores() {
		double[] scArray = new double[population];
		
		for (int i=0; i<population; i++) {
			for (int j=0; j<protoScore; j++) {
				if (speciesArray[i].toCharArray()[j] == prototype.toCharArray()[j]) {
					scArray[i] += 1;
				}
				scArray[i] = formula(scArray[i]);
			}
		}		
		
		double sumArray = Arrays.stream(scArray).sum();
		
		for (int i=0; i<population; i++) {
			scArray[i] = scArray[i] / sumArray;
		}
		return scArray;
	}
	
	
	public String procreate(String evolType, int k, double mutaProb) {
		ArrayList<String> contestants = new ArrayList<String>();
		int sum = 0;
		int s = (int) Math.ceil(protoScore/2);
		String child = "";
		
		if (k != 2) {
			evolType = "rand";
		}
		
		do {
			for (int i=0; i<population; i++) {
				if (scoreArray[i] > Math.random() * (1 - 0)){
					contestants.add(speciesArray[i]);
					sum +=1;
				if (sum == k) {
					break;
					}
				}
			}
		} while(sum < k);

		if (evolType == "rand") {
			for (int i=0; i<protoScore; i++) {	
				int mask = (int) (Math.floor(Math.random() * (k - 0)));
				if ((Math.random() * (1 - 0)) <= mutaProb) {
					child += (char) (Math.floor(Math.random() * (126 - 32)) + 32);
				} else {
					child += contestants.get(mask).charAt(i);				
				}
			}
		} else if (evolType == "split") {
			child += contestants.get(0).substring(0, s);
			child += contestants.get(1).substring(s, protoScore);

			for (int i=0; i<protoScore; i++) {
				if ((Math.random() * (1 - 0)) <= mutaProb) {
					child.toCharArray()[i] = (char) (Math.floor(Math.random() * (126 - 32)) + 32);
				}
			}
		}
		
		return child;
	}
	
	
	public experiment(int createPopulation, String definePrototype) {
	    population = createPopulation;
	    prototype = definePrototype;
	    protoScore = definePrototype.length();
	    speciesArray = generatePopulation();
	    scoreArray = calculateScores();
	}
	
	
	public static void main(String[] args) {
		experiment stringExp = new experiment(100, "The Orville!");
		while (stringExp.paragon().equals(stringExp.prototype) == false) {
			stringExp.generation += 1;
			for (int species=0; species<stringExp.population; species++) {
				stringExp.speciesArray[species] = stringExp.procreate("rand", 2, 0.01);
			}
			stringExp.scoreArray = stringExp.calculateScores();
			if (stringExp.generation % 10 == 0) {
				System.out.println(stringExp.generation + ". :" + stringExp.paragon());				
			}
		}
		System.out.println("Final: " + stringExp.generation + " " + stringExp.paragon());
	}
}
