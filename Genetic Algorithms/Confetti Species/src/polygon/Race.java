package polygon;

import java.util.ArrayList;
import java.util.Arrays;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JPanel;
import javax.swing.Timer;
import javax.swing.JFrame;

public class Race extends JPanel implements ActionListener{
	
	private static final long serialVersionUID = 1L;
	
	public int tframe = 5, round = 3900, pop = 1000, k = 2; 
	/* tframe <- Timer delay milliseconds
	 * round <- direction vector/array length / how long the speciment is alive
	 * pop <- number of speciment in the population
	 * k <- number of parents used for reproduction
	 */
	public final int size_x = 1366, size_y = 768;
	// window size variables
	double mutaProb = 0.025; 
	// mutaProb <- chance for random mutation
	public int generation = 1;
	// generation <- generation counter
	public final int tx = 1300, ty = 384;
	// tx, ty <- target x, y coordinates
	public int i = 0;
	// i <- iteration, round, lifetime counter
	public speciment[] population = new speciment[pop];
	// population <- population size
	public double[] scoreArray = new double[pop];
	// scoreArray <- score array for selection parents
	
	public Timer t = new Timer(tframe, this);
	// Timer <- timer for iterations

	public class speciment {
		/*speciment class
		 * x, y <- coordinates of the speciement 
		 * velxvec, velyvec <- direction of the speciment on x and y axes
		 * rC, gC, bC <- color variables for RGB speciment colour 'c'
		 * alive <- whether the speciment is alive or dead
		 */
		public int x = 166, y = 384;
		public int[] velxvec = new int[round];
		public int[] velyvec = new int[round];
		int rC = (int) (Math.floor(Math.random() * 255));
		int gC = (int) (Math.floor(Math.random() * 255));
		int bC = (int) (Math.floor(Math.random() * 255));
		public Color c = new Color (rC, gC, bC);
		public boolean alive = true;
	}
	
	public void commonSense(int speciment, int step) {
		/* Method for correcting the directions to avoid back and forth movement of the speciment 
		 *  e.g. change Array{-1, 1, -1, 1, 0, 0, -1} to Array{-1, -1, -1, -1, 0, 0, -1}
		 */
		if (step > 0) {
			if (population[speciment].velxvec[step] * -1 == population[speciment].velxvec[step-1]) {
				population[speciment].velxvec[step] = population[speciment].velxvec[step] * -1;
			}
			if (population[speciment].velyvec[step] * -1 == population[speciment].velyvec[step-1]) {
				population[speciment].velyvec[step] = population[speciment].velyvec[step] * -1;
			}
		}
	}
	
	public void fill() {
		/* Method for filling the velyvec & velxvec of each speciment with random numbersof {-1, 0, 1}
		 *  p <- speciment in population
		 *  s <- step / cell of the arrays 
		 */ 
		for (int p=0; p<pop; p++) { // iterate over the population array
			population[p] = new speciment(); // create 'pop' number of speciments
			
			for (int s=0; s<round; s++) { // fill in the velxvec, velyvec arrays with random integers [-1, 1]
				population[p].velxvec[s] = (int) (Math.floor(Math.random() * (3) - 1));
				population[p].velyvec[s] = (int) (Math.floor(Math.random() * (3) - 1));
				commonSense(p, s);
			}
		}
	}

	public void calculateScores() {
		/* Method for calculating the relative probabilties for selection based on the fittness scores of each speciment
		 * p <- speciment in poulation
		 */
		for (int p=0; p<pop; p++) { // Calculate the distance measure fo each species
			if (population[p].alive == true){
				scoreArray[p] = Math.sqrt(Math.pow((tx-population[p].x), 2) + Math.pow((ty-population[p].y), 2));
			}
		}
		
		double maxArray = Arrays.stream(scoreArray).max().getAsDouble(); // get the farthest/worst of the species
		
		for (int p=0; p<pop; p++) { // calculate the fitness scores of each speciment
			if (population[p].alive == true){
				scoreArray[p] = -Math.log(Math.pow(scoreArray[p] / maxArray, 8));
			} else {
				scoreArray[p] = 0;
			}
		}
		
		double sumArray = Arrays.stream(scoreArray).sum(); // get the sum of the score array / total fittness scores
		for (int p=0; p<pop; p++) { // calculate the relative probabilities for selection
			scoreArray[p] = scoreArray[p] / sumArray;
		}
	}
	
	public void evolve(){
		/* Method for selecting parents based on relative probability and mutating parts of them randomly based on a probability
		* child <- speciments in population for the next generation
		* parent <- speciements in the current population
		*/
		for (int child=0; child<pop; child++) { // iterate over all the speciments in population
			population[child].x = 166; // set x variable to starting point
			population[child].y = 384; // set y variable to starting point
			population[child].alive = true; // bring the dead speciments back to life

			ArrayList<speciment> contestants = new ArrayList<speciment>(); //Arraylist for potential parents
			int count = 0; // number of speciments for potential parents
			
			do { // keep selecting potential parents until the specified 'k' number of speciements are selected
				for (int parent=0; parent<pop; parent++) { // iterate over all of the parents in order to give every speciment a fair chance for selection
					if (scoreArray[parent] > Math.random() * (1 - 0)){ // select/don't select a parent based on its relative probability
						contestants.add(population[parent]);
						count +=1;
					}
				}
			} while(count < k);
		
			for (int s=0; s<round; s++) { // iterate over the direction vectors of each children	
				
				if ((Math.random() * (1 - 0)) <= mutaProb) { // mutate or inherit the specific gene based on the 'mutaProb' variable
					population[child].velxvec[s] = (int) (Math.floor(Math.random() * (3) - 1));
					population[child].velyvec[s] = (int) (Math.floor(Math.random() * (3) - 1));
				} else { // the child inherits the 's' gene/direction of the 'mask' parent in the aray list
					int mask = (int) (Math.floor(Math.random() * (k - 0))); 
					population[child].velxvec[s] = contestants.get(mask).velxvec[s];				
					population[child].velyvec[s] = contestants.get(mask).velyvec[s];
				}
				commonSense(child, s); // check for back and forth movement and correct it
			}
			
			population[child].rC = contestants.get((int) (Math.floor(Math.random() * (k - 0)))).rC; // randomly update red color
			population[child].gC = contestants.get((int) (Math.floor(Math.random() * (k - 0)))).gC; // randomly update green color
			population[child].bC = contestants.get((int) (Math.floor(Math.random() * (k - 0)))).bC; // randomly update blue color
			
		}
	}
	
	public void paintComponent(Graphics g) {
		/* Method for displaying the initial population and starting the timer
		 * p <- speciment
		 */
		super.paintComponent(g);
		
		g.drawOval(tx-10, ty-10, 20, 20); // draw target, towards which the speciments are going
		
		for (int p=0; p<pop; p++) { // draw the population
			g.setColor(population[p].c); // set color
			g.fillOval(population[p].x, population[p].y, 10, 10); // draw the ovals			
		}	
		t.start(); //start timer
	}
	
	public void actionPerformed(ActionEvent e) {
		/* Method for moving the speciments one step
		 *  p <- speciment in the population
		 */
		if (i == round){ // if the iterations reach the end of the lifecycle (round) perform the evolution
			i = 0; // set lifecycle variable to 0
			generation += 1; // next generation
			calculateScores();
			evolve();

		} else { 
			if (i % round == 0) { // print generation umbers
				System.out.println("Generation: " + generation);
			}

			for (int p=0; p<pop; p++) { // move the speciment p based on its i-th direction
				if (population[p].alive == true) { // check for life-signes
					population[p].x += population[p].velxvec[i]; // move the speciment p on its x values
					population[p].y += population[p].velyvec[i]; // move the speciment p on its y values
					if (population[p].x >= (size_x+100) || population[p].x <= 0) { // check to see if the speciment has reached the edge of the windows
						population[p].alive = false; // if yes, make it dead
					}
					if (population[p].y >= (size_y-50) || population[p].y <= 0) { // check to see if the speciment has reached the edge of the windows
						population[p].alive = false; // if yes, make it dead
					}
				}
				if(population[p].x > tx-10 && population[p].x < tx+10 && population[p].y > ty-10 && population[p].y < ty+10) {
					// check if the speciment is within the target area
					System.out.println("---------------------------");
					System.out.println("Final Generation: " + generation);
					System.out.println("Population: " + pop);
					System.out.println("Lifecycle size: " + round);
					System.out.println("Number of Parents: " + k);
					System.out.println("Mutation Probability: " + (int) (mutaProb * 100) + "%");

					System.exit(0);
				}
			}
							
			i += 1; // increment iteration variable / lifecycle
			repaint(); // re-draw the population on the new coordinates

		}	
	}
	
	
	public static void main(String[] args) {
		Race r = new Race(); // create new Race
		r.fill(); // fill in the initial values
		JFrame jf = new JFrame(); // new JFrame
		
		jf.setTitle("Evolution"); // set title
		jf.setSize((r.size_x), (r.size_y)); // set size
		jf.setVisible(true); // set visibility to true
		jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // set operation for close
		jf.add(r); // add class to JFrame

	}
		
}
