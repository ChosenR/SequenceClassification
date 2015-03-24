package weka.classifiers.trees.shapelet_tree;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.TreeMap;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.Shapelet;
import weka.core.shapelet.OrderLineObj;

public class ShapeletTreeClassifierModified extends Classifier {

	private static final long serialVersionUID = 1L;
	private ShapeletNode root;
	private static String logFileName;
	private int minLength, maxLength;

	// constructors
	public ShapeletTreeClassifierModified(String logFileName) throws Exception {
		this.root = new ShapeletNode();
		ShapeletTreeClassifierModified.logFileName = logFileName;
		minLength = maxLength = 0;
		FileWriter fw = new FileWriter(logFileName);
		fw.close();
	}

	public void setShapeletMinMaxLength(int minLength, int maxLength) {
		this.minLength = minLength;
		this.maxLength = maxLength;
	}

	public void buildClassifier(ArrayList<Instances> datasets) throws Exception {
		if (minLength < 1 || maxLength < 1) {
			throw new Exception(
					"Shapelet minimum or maximum length is incorrectly specified!");
		}

		root.trainShapeletTree(datasets, minLength, maxLength, 0);
	}

	public double classifyInstance(Instance instance) {
		return root.classifyInstance(instance);
	}

	// Shapelet Node
	private class ShapeletNode implements Serializable {

		private static final long serialVersionUID = 1L;
		private ShapeletNode leftNode;
		private ShapeletNode rightNode;
		private double classDecision;
		private Shapelet shapelet;


		public ShapeletNode() {
			leftNode = null;
			rightNode = null;
			classDecision = -1;
		}

		public void trainShapeletTree(ArrayList<Instances> datasets,
				int minShapeletLength, int maxShapeletLength, int level)
				throws Exception {

			FileWriter fw = new FileWriter(logFileName, true);
			fw.append("level:" + level + "," + "\n");
			for (int i = 0; i < datasets.size(); i++) {
				fw.append("Num of Instances from " + i + " dataset:"
						+ datasets.get(i).numInstances() + "\n");
			}
			for (int i = 0; i < datasets.size(); i++) {
				System.out.println("Num of Instances from dataset: " + i + "-->"
						+ datasets.get(i).numInstances() + "\n");
			}
			fw.close();

			// ----------------------------------------------------------------------------------//
			// 1. check whether this is a leaf node with only one class present
			// - base case
			// MODIFICATION - Each node has different datasets, but all of them hold the same instances but in different granularities
			// If one of them has only one class present - base case
			boolean oneClass = true;
			double firstClassValue = -1.0;

			System.out.println("*****Check if this node is in base case.");
			firstClassValue = datasets.get(0).instance(0).classValue();
			oneClass = true;
			for (int i = 1; i < datasets.get(0).numInstances(); i++) {
				if (datasets.get(0).instance(i).classValue() != firstClassValue) {
					oneClass = false;
					break;
				}
			}

			if (oneClass == true) {
				this.classDecision = firstClassValue; // no need to find shapelet, base  case
				System.out.println("Found Leaf! Class decision: " + firstClassValue);
				fw = new FileWriter(logFileName, true);
				fw.append("FOUND LEAF --> class decision here: "
						+ firstClassValue + "\n" );
				fw.close();
			}



			else {

				// ----------------------------------------------------------------------------------//

				try {
					// 2. find the best shapelet to split the data
					fw = new FileWriter(logFileName, true);
					fw.append("--> 1.Find the best shapelet to split the data!"
							+ "\n");
					fw.close();

					this.shapelet = findBestShapelet(datasets,
							minShapeletLength, maxShapeletLength);
					System.out.println("Best Shapelet");
					printShapelet(this.shapelet);

					// ----------------------------------------------------------------------------------//
					// 3. split the data in every dataset using the shapelet and
					// create new data sets

					ArrayList<Instances> leftInstancesAggr = new ArrayList<Instances>();
					ArrayList<Instances> rightInstancesAggr = new ArrayList<Instances>();
					
					leftNode = new ShapeletNode();
					rightNode = new ShapeletNode();

					Shapelet the_shapelet = null;
					
					// Find the dataset with the same granularity as the shapelet.
					// Then, compute the subsequence distance and check if it goes to left or right node
					// Finally, move the corresponding instances with different granularity to the same side.
					for (int z = 0; z < datasets.size(); z++) {
						
						System.out.println("Split dataset :" + z);
						
						// find the corresponding shapelet with different granularity
						if(shapelet.granularity == z){
						the_shapelet = this.shapelet; 
						System.out.println("Can use this shapelet to divide!");
						
						
						

						double dist;											
						ArrayList<ArrayList<Instance>> leftSplit = new ArrayList<ArrayList<Instance>>(datasets.size());
						ArrayList<ArrayList<Instance>> rightSplit = new ArrayList<ArrayList<Instance>>(datasets.size());
					    
						//init the structures
					    for(int i=0; i<datasets.size();i++){
					    	ArrayList<Instance> l = new ArrayList<Instance>();  
					    	ArrayList<Instance> r = new ArrayList<Instance>();  
							leftSplit.add(l);
							rightSplit.add(r);
					    }
					    
					    
					    
					    fw = new FileWriter(logFileName, true);
						fw.append("-->2. split the data using the shapelet and create new data sets");
						fw.close();

						for (int i = 0; i < datasets.get(z).numInstances(); i++) {
							dist = subsequenceDistance( the_shapelet.getContent(), datasets.get(z).instance(i).toDoubleArray());
							System.out.println("dist:" + dist);
							if (dist < the_shapelet.getSplitThreshold()) {
								
								//That specific instance in all granularities is going to the left node
								for(int x=0; x<datasets.size();x++){
									leftSplit.get(x).add(datasets.get(x).instance(i));									
								}
								System.out.println("gone left");
								
							} else {
								//That specific instance in all granularities is going to the left node
								for(int x=0; x<datasets.size();x++){
									rightSplit.get(x).add(datasets.get(x).instance(i));									
								}
								System.out.println("gone right");
							}
						}
						System.out.println("leftSize:" + leftSplit.get(z).size());	
						System.out.println("rightSize:" + rightSplit.get(z).size());

						// ----------------------------------------------------------------------------------//

						// 4. initialise and recursively compute children nodes
					

	
						// MODIFICATION - Now each node can hold more than one
						// set of instances, depending of the number of
						// granularities
					
							for(int v=0; v<leftSplit.size();v++){
								Instances l = new Instances(datasets.get(v), leftSplit.get(v).size());
								for (int i = 0; i < leftSplit.get(v).size(); i++) {
									l.add(leftSplit.get(v).get(i));
								}
								leftInstancesAggr.add(l);				
							}					
						
							for(int v=0; v<rightSplit.size();v++){
								Instances r = new Instances(datasets.get(v), rightSplit.get(v).size());
								for (int i = 0; i < rightSplit.get(v).size(); i++) {
									r.add(rightSplit.get(v).get(i));
								}
								rightInstancesAggr.add(r);				
							}
					}
					}
		

					fw = new FileWriter(logFileName, true);
					for (int s = 0; s < datasets.size(); s++) {
						fw.append("left size under level " + level + ": "
								+ leftInstancesAggr.get(s).numInstances()
								+ "\n");
					}
					fw.close();
					leftNode.trainShapeletTree(leftInstancesAggr,minShapeletLength, maxShapeletLength, (level + 1));

					fw = new FileWriter(logFileName, true);
					for (int s = 0; s < datasets.size(); s++) {
						fw.append("right size under level " + level + ": "
								+ rightInstancesAggr.get(s).numInstances()
								+ "\n");
					}
					fw.close();

					rightNode.trainShapeletTree(rightInstancesAggr, minShapeletLength, maxShapeletLength, (level + 1));

				} catch (Exception e) {
					System.out.println("Problem initialising tree node: " + e);
					e.printStackTrace();
				}
			}
		}

		
		//TO DO: find the corresponding shapelet. 
		//Find the same shapelet but with different granularity, bellonging to another dataset
		
		/*private Shapelet findCorrespondingShapelet(Shapelet shapelet, int granularity, ArrayList<Instances> datasets) throws IOException {
			
			TreeMap<Double, Integer> classDistribution = getClassDistributions(datasets
					.get(granularity)); // used to compute gain ratio
		
			
			
			double[] wholeInstance = datasets.get(granularity).instance(shapelet.seriesId).toDoubleArray();
			double[] newContent = new double[shapelet.content.length];
			
			for (int m = shapelet.startPos; m < shapelet.startPos+ shapelet.content.length; m++) {
				newContent[m - shapelet.startPos] = wholeInstance[m];
			}

			newContent = zNorm(newContent, false);
			
			Shapelet newShapelet = checkCandidate(newContent, datasets.get(granularity), shapelet.seriesId, shapelet.startPos,
					classDistribution, granularity);
			
			System.out.println("The Shapelet");
			printShapelet(shapelet);
			
			System.out.println("The corresponding Shapelet");
			printShapelet(newShapelet);

			
			return newShapelet;
			}*/

		
		//TO DO: Modify this function to take into account the different granularities 
		public double classifyInstance(Instance instance) {
			if (this.leftNode == null) {
				return this.classDecision;
			} else {
				double distance;
				distance = subsequenceDistance(this.shapelet.getContent(),
						instance, false);

				if (distance < this.shapelet.getSplitThreshold()) {
					return leftNode.classifyInstance(instance);
				} else {
					return rightNode.classifyInstance(instance);
				}
			}
		}
	}


	
	

	private Shapelet findBestShapelet(ArrayList<Instances> datasets,
			int minShapeletLength, int maxShapeletLength) throws IOException {

		Shapelet bestShapelet = null;

		// for all datasets in the different levels of granularity
		for (int j = 0; j < datasets.size(); j++) {

			TreeMap<Double, Integer> classDistributions = getClassDistributions(datasets
					.get(j)); // used to compute gain ratio
			System.out.println("Processing data in dataset: " + j);

			// for all time series in that dataset
			for (int i = 0; i < datasets.get(j).numInstances(); i++) {
				System.out.println((1 + i) + "/"
						+ datasets.get(j).numInstances() + "\t Started: "
						+ getTime());

				double[] wholeCandidate = datasets.get(j).instance(i)
						.toDoubleArray();

				// for all lengths
				for (int length = minShapeletLength; length <= maxShapeletLength; length++) {

					// for all possible starting positions of that length
					for (int start = 0; start <= wholeCandidate.length - length
							- 1; start++) {

						// CANDIDATE ESTABLISHED - got original series, length
						// and starting position
						// extract relevant part into a double[] for processing
						double[] candidate = new double[length];
						for (int m = start; m < start + length; m++) {
							candidate[m - start] = wholeCandidate[m];
						}

						candidate = zNorm(candidate, false);
						//System.out.println("CheckCandidate: " + j);
						Shapelet candidateShapelet = checkCandidate(candidate,
								datasets.get(j), i, start, classDistributions,
								j);
						
						//appendShapelet(candidateShapelet);

						if (bestShapelet == null
								|| candidateShapelet.compareTo(bestShapelet) < 0) {
							bestShapelet = candidateShapelet;

						}
					}
				}
			}
		}
		return bestShapelet;
	}

	private static Shapelet checkCandidate(double[] candidate, Instances data,
			int seriesId, int startPos,
			TreeMap<Double, Integer> classDistribution, int granularity) throws IOException {

		// create orderline by looping through data set and calculating the
		// subsequence distance from candidate to all data, inserting in order.
		ArrayList<OrderLineObj> orderline = new ArrayList<OrderLineObj>();

		for (int i = 0; i < data.numInstances(); i++) {
			double distance = subsequenceDistance(candidate, data.instance(i),
					true);
			double classVal = data.instance(i).classValue();

			boolean added = false;
			// add to orderline
			if (orderline.isEmpty()) {
				orderline.add(new OrderLineObj(distance, classVal));
				added = true;
			} else {
				for (int j = 0; j < orderline.size(); j++) {
					if (added == false
							&& orderline.get(j).getDistance() > distance) {
						orderline.add(j, new OrderLineObj(distance, classVal));
						added = true;
					}
				}
			}
			// if obj hasn't been added, must be furthest so add at end
			if (added == false) {
				orderline.add(new OrderLineObj(distance, classVal));
			}
		}

		appendOrderLine(orderline);


		// create a shapelet object to store all necessary info, i.e.
		// content, seriesId, then calc info gain, split threshold
		Shapelet shapelet = new Shapelet(candidate, seriesId, startPos,
				granularity);
		shapelet.calcGainRatioAndThreshold(orderline, classDistribution);

		return shapelet;
	}

	public static double subsequenceDistance(double[] candidate,
			Instance timeSeriesIns, boolean earlyAbandon) {
		double[] timeSeries = timeSeriesIns.toDoubleArray();
		if (earlyAbandon)
			return subsequenceDistanceEarlyAbandon(candidate, timeSeries);
		else
			return subsequenceDistance(candidate, timeSeries);

	}

	// modification to add early abandon!
	public static double subsequenceDistanceEarlyAbandon(double[] candidate,
			double[] timeSeries) {

		double bestSum = Double.MAX_VALUE;
		double sum = 0;
		double[] subseq;
		boolean stop = false;

		// for all possible subsequences
		for (int i = 0; i <= timeSeries.length - candidate.length - 1; i++) {
			sum = 0;
			stop = false;
			// get subsequence of T that is the same lenght as candidate
			subseq = new double[candidate.length];

			for (int j = i; j < i + candidate.length; j++) {
				subseq[j - i] = timeSeries[j];
			}
			subseq = zNorm(subseq, false); // Z-NORM HERE

			for (int j = 0; j < candidate.length; j++) {
				sum += (candidate[j] - subseq[j]) * (candidate[j] - subseq[j]);

				// early abandon -> We don't need do complete the computation,
				// as soon as we reach a higher distance than the best one found
				// so far
				if (sum >= bestSum) {
					stop = true;
					break;
				}
			}
			if (!stop) {
				bestSum = sum;
			}
		}
		return (1.0 / candidate.length * bestSum);
	}

	public static double subsequenceDistance(double[] candidate,
			double[] timeSeries) {

		// double[] timeSeries = timeSeriesIns.toDoubleArray();
		double bestSum = Double.MAX_VALUE;
		double sum = 0;
		double[] subseq;

		// for all possible subsequences of two
		for (int i = 0; i <= timeSeries.length - candidate.length - 1; i++) {
			sum = 0;
			// get subsequence of two that is the same lenght as one
			subseq = new double[candidate.length];

			for (int j = i; j < i + candidate.length; j++) {
				subseq[j - i] = timeSeries[j];
			}
			subseq = zNorm(subseq, false); // Z-NORM HERE
			for (int j = 0; j < candidate.length; j++) {
				sum += (candidate[j] - subseq[j]) * (candidate[j] - subseq[j]);
			}
			if (sum < bestSum) {
				bestSum = sum;
			}
		}
		return (1.0 / candidate.length * bestSum);
	}

	public static double[] zNorm(double[] input, boolean classValOn) {
		double mean;
		double stdv;

		double classValPenalty = 0;
		if (classValOn) {
			classValPenalty = 1;
		}
		double[] output = new double[input.length];
		double seriesTotal = 0;

		for (int i = 0; i < input.length - classValPenalty; i++) {
			seriesTotal += input[i];
		}

		mean = seriesTotal / (input.length - classValPenalty);
		stdv = 0;
		for (int i = 0; i < input.length - classValPenalty; i++) {
			stdv += (input[i] - mean) * (input[i] - mean);
		}

		stdv = stdv / input.length - classValPenalty;
		stdv = Math.sqrt(stdv);

		for (int i = 0; i < input.length - classValPenalty; i++) {
			output[i] = (input[i] - mean) / stdv;
		}

		if (classValOn == true) {
			output[output.length - 1] = input[input.length - 1];
		}

		return output;
	}



	private static TreeMap<Double, Integer> getClassDistributions(Instances data) {
		TreeMap<Double, Integer> classDistribution = new TreeMap<Double, Integer>();
		double classValue;
		for (int i = 0; i < data.numInstances(); i++) {
			classValue = data.instance(i).classValue();
			boolean classExists = false;
			for (Double d : classDistribution.keySet()) {
				if (d == classValue) {
					int temp = classDistribution.get(d);
					temp++;
					classDistribution.put(classValue, temp);
					classExists = true;
				}
			}
			if (classExists == false) {
				classDistribution.put(classValue, 1);
			}
		}
		return classDistribution;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		// TODO Auto-generated method stub

	}

	
	
	/*
	 * Auxiliar Functions
	 */
	
	public static String getTime() {
		Calendar calendar = new GregorianCalendar();
		return calendar.get(Calendar.DAY_OF_MONTH) + "/"
				+ calendar.get(Calendar.MONTH) + "/"
				+ calendar.get(Calendar.YEAR) + " - "
				+ calendar.get(Calendar.HOUR_OF_DAY) + ":"
				+ calendar.get(Calendar.MINUTE) + ":"
				+ calendar.get(Calendar.SECOND) + " AM";
	}
	
	
	public void printShapelet(Shapelet shapelet){
		System.out.println("----------------------");
		System.out.println("Gain Ratio:" + shapelet.getGainRatio());
		System.out.println("Information Gain:" + shapelet.getInformationGain() );
		System.out.println("Split Info:" + shapelet.getSplitInfo() );
		System.out.println("SplitPoint:" + shapelet.splitThreshold );
		System.out.println("length:" + shapelet.getLength() );
		System.out.println("SeriesId: " + shapelet.getSeriesId() );
		System.out.println("startPosition: " + shapelet.getStartPos() );
		System.out.println("Found in dataset: " + shapelet.granularity);
		System.out.println("--------------------------------------");
	}
	
	public void appendShapelet(Shapelet shapelet) throws IOException{
		FileWriter fw = new FileWriter(logFileName, true);
		fw.append("----------------- + \n");
		fw.append("SeriesID: "  + shapelet.getSeriesId() + "," + "\n"+ 
				  "StartPos: "  + shapelet.getStartPos() + "," + "\n"+
				  "Lenght: "    + shapelet.getContent().length + "," + "\n"+
				  "GainRatio: " + shapelet.getGainRatio() + "," + "\n"+
				  "InfoGain: "  + shapelet.getInformationGain() + "," + "\n"+
				  "SplitInfo: " + shapelet.getSplitInfo() + "," + "\n" +
				  "Threshold: " + shapelet.splitThreshold + "," + "\n"+
				  "DataSet: " 	+ shapelet.granularity + "," + "\n");
		fw.close();
	}
	
	
	public static void appendOrderLine(ArrayList<OrderLineObj> orderline) throws IOException{
		 
		FileWriter fw = new FileWriter(logFileName, true);
		fw.append("---->OrderLine<---- + \n");
		
		for(int x =0; x<orderline.size();x++)
		  {
			  fw.append(orderline.get(x).getDistance() + "-->" +
					  orderline.get(x).getClassVal() + "\n");
			}
		 
		fw.close();
	}
}
