    package weka.core.shapelet;

import java.util.ArrayList;
import java.util.TreeMap;


 
    //shapelet Class
	public class Shapelet implements Comparable<Shapelet> {
		

		public double[] content; 
		public int seriesId;
		public int startPos;
		public double splitThreshold;
		public double informationGain;
		public double separationGap;
		private double gainRatio;

		
		//Constructors
		public Shapelet(double[] content, int seriesId, int startPos) {
			this.setContent(content);
			this.setSeriesId(seriesId);
			this.setStartPos(startPos);
		}

		public Shapelet(double[] content, int seriesId, int startPos,
				double splitThreshold, double gain, double gap) {
			this.setContent(content);
			this.setSeriesId(seriesId);
			this.setStartPos(startPos);
			this.setSplitThreshold(splitThreshold);
			this.setInformationGain(gain);
			this.setSeparationGap(gap) ;
		}


		//Getters and Setters
		public double[] getContent() {
			return content;
		}

		public void setContent(double[] content) {
			this.content = content;
		}

		public int getSeriesId() {
			return seriesId;
		}

		public void setSeriesId(int seriesId) {
			this.seriesId = seriesId;
		}

		public int getStartPos() {
			return startPos;
		}

		public void setStartPos(int startPos) {
			this.startPos = startPos;
		}

		public double getSplitThreshold() {
			return splitThreshold;
		}

		public void setSplitThreshold(double splitThreshold) {
			this.splitThreshold = splitThreshold;
		}

		public double getInformationGain() {
			return informationGain;
		}

		public void setInformationGain(double informationGain) {
			this.informationGain = informationGain;
		}

		public double getSeparationGap() {
			return separationGap;
		}

		public void setSeparationGap(double separationGap) {
			this.separationGap = separationGap;
		}

		
		public double getGainRatio (){
			return this.gainRatio;
		
		}
		
		private void setGainRatio(double bsfGainR) {
			this.gainRatio = bsfGainR;
			
		}
		public int getLength(){
			return content.length;
		}
		

		/*
		 * Compute Information Gain
		 * 1 - For each threshold (starting between 0 and 1 and ending between end-1 and end
		 * 2 - Compute the information gain (Parent Entropy - EntropyAfterSplit)
		 * 3 - EntropyAfterSplit = EntropyLeft + EntropyRight
		 */
		public void calcInfoGainAndThreshold(
				ArrayList<OrderLineObj> orderline,
				TreeMap<Double, Integer> classDistribution) {
			
			
			double lastDist = orderline.get(0).getDistance(); 
			double thisDist = -1;
			double bsfGain = -1;
			double threshold = -1;

			for (int i = 1; i < orderline.size(); i++) {
				thisDist = orderline.get(i).getDistance();
				if (i == 1 || thisDist != lastDist) { // check that threshold has moved

					// count class instances below and above threshold
					TreeMap<Double, Integer> lessClasses = new TreeMap<Double, Integer>();
					TreeMap<Double, Integer> greaterClasses = new TreeMap<Double, Integer>();

					for (double j : classDistribution.keySet()) {
						lessClasses.put(j, 0);
						greaterClasses.put(j, 0);
					}

					int sumOfLessClasses = 0;
					int sumOfGreaterClasses = 0;

					// visit those below threshold
					for (int j = 0; j < i; j++) {
						double thisClassVal = orderline.get(j).getClassVal();
						int storedTotal = lessClasses.get(thisClassVal);
						storedTotal++;
						lessClasses.put(thisClassVal, storedTotal);
						sumOfLessClasses++;
					}

					// visit those above threshold
					for (int j = i; j < orderline.size(); j++) {
						double thisClassVal = orderline.get(j).getClassVal();
						int storedTotal = greaterClasses.get(thisClassVal);
						storedTotal++;
						greaterClasses.put(thisClassVal, storedTotal);
						sumOfGreaterClasses++;
					}

					int sumOfAllClasses = sumOfLessClasses
							+ sumOfGreaterClasses;

					double parentEntropy = entropy(classDistribution);

					// calculate the info gain below the threshold
					double lessFrac = (double) sumOfLessClasses
							/ sumOfAllClasses;
					double entropyLess = entropy(lessClasses);
					// calculate the info gain above the threshold
					double greaterFrac = (double) sumOfGreaterClasses
							/ sumOfAllClasses;
					double entropyGreater = entropy(greaterClasses);

					double gain = parentEntropy - lessFrac * entropyLess
							- greaterFrac * entropyGreater;

					if (gain > bsfGain) {
						bsfGain = gain;
						threshold = (thisDist - lastDist) / 2 + lastDist;
					}
				}
				lastDist = thisDist;
			}
			if (bsfGain >= 0) {
				this.setInformationGain(bsfGain);
				this.splitThreshold = threshold;
			}
		}
		
		
		

		/*
		 * Compute Gain Ratio: Information Gain / Split Info
		 * 1 - For each threshold (starting between 0 and 1 and ending between end-1 and end
		 * 2 - Compute the information gain (Parent Entropy - EntropyAfterSplit)
		 * 3 - EntropyAfterSplit = EntropyLeft + EntropyRight
		 */
		public void calcGainRatioAndThreshold(
				ArrayList<OrderLineObj> orderline,
				TreeMap<Double, Integer> classDistribution) {
			
			
			
			double lastDist = orderline.get(0).getDistance(); 
			double thisDist = -1;
			double bsfGainR = -1;
			double bsfGain = -1;
			@SuppressWarnings("unused")
			double threshold = -1;
			double Infogain = 0;

			for (int i = 1; i < orderline.size(); i++) {
				thisDist = orderline.get(i).getDistance();
				if (i == 1 || thisDist != lastDist) { // check that threshold has moved

					// count class instances below and above threshold
					TreeMap<Double, Integer> lessClasses = new TreeMap<Double, Integer>();
					TreeMap<Double, Integer> greaterClasses = new TreeMap<Double, Integer>();

					for (double j : classDistribution.keySet()) {
						lessClasses.put(j, 0);
						greaterClasses.put(j, 0);
					}

					int sumOfLessClasses = 0;
					int sumOfGreaterClasses = 0;

					// visit those below threshold
					for (int j = 0; j < i; j++) {
						double thisClassVal = orderline.get(j).getClassVal();
						int storedTotal = lessClasses.get(thisClassVal);
						storedTotal++;
						lessClasses.put(thisClassVal, storedTotal);
						sumOfLessClasses++;
					}

					// visit those above threshold
					for (int j = i; j < orderline.size(); j++) {
						double thisClassVal = orderline.get(j).getClassVal();
						int storedTotal = greaterClasses.get(thisClassVal);
						storedTotal++;
						greaterClasses.put(thisClassVal, storedTotal);
						sumOfGreaterClasses++;
					}

					int sumOfAllClasses = sumOfLessClasses
							+ sumOfGreaterClasses;

					double parentEntropy = entropy(classDistribution);

					// calculate the info gain below the threshold
					double lessFrac = (double) sumOfLessClasses
							/ sumOfAllClasses;
					double entropyLess = entropy(lessClasses);
					// calculate the info gain above the threshold
					double greaterFrac = (double) sumOfGreaterClasses
							/ sumOfAllClasses;
					double entropyGreater = entropy(greaterClasses);

					Infogain = parentEntropy - lessFrac * entropyLess
							- greaterFrac * entropyGreater;


					double splitInfo = splitInfo(orderline);
						
					
					gainRatio = Infogain/splitInfo;
	
					
					
					if (gainRatio > bsfGainR) {
						bsfGainR = gainRatio;
						threshold = (thisDist - lastDist) / 2 + lastDist;
					}
					
					if (Infogain > bsfGain) {
						bsfGain = Infogain;
					}
				}
				lastDist = thisDist;
			}
			if (bsfGainR >= 0) {
				this.setGainRatio(bsfGainR);
			}
			if (bsfGain >= 0) {
				this.setInformationGain(bsfGain);
			}
		}
	
	
	
	

	//Compute SplitInfo
	public double splitInfo(ArrayList<OrderLineObj> orderline){
		ArrayList<Double> splitInfoParts = new ArrayList<Double>();
		double toAdd;
		double thisPart=0;	
		double splitInfo = 0;
		double totalElements = orderline.size();
		double sum=0;
		double distToCompare = orderline.get(0).getDistance();
		
		for(int x=0; x<orderline.size();x++){
			if(orderline.get(x).getDistance() == distToCompare){
				sum++;			
			}
			else{
				thisPart =  (sum / totalElements);
				toAdd = - thisPart * Math.log10(thisPart) / Math.log10(2);
					if (Double.isNaN(toAdd))
					toAdd = 0;
				splitInfoParts.add(toAdd);		
				sum = 1;	
				toAdd=0;
			}
			
			distToCompare = orderline.get(x).getDistance();	
		}
		
		if(sum>0){
			thisPart = (sum/totalElements);
			toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
				if (Double.isNaN(toAdd))
				toAdd = 0;
			splitInfoParts.add(toAdd);	
			toAdd=0;
		}
		

		for (int i = 0; i < splitInfoParts.size(); i++) {
			splitInfo += splitInfoParts.get(i);
		}

		return splitInfo;
		
	}

	
	
	
	
	
	
	
	
	
	
		//Compute Entropy
		private static double entropy(TreeMap<Double, Integer> classDistributions) {
			if (classDistributions.size() == 1) {
				return 0;
			}

			double thisPart;
			double toAdd;
			int total = 0;
			for (Double d : classDistributions.keySet()) {
				total += classDistributions.get(d);
			}
			// to avoid NaN calculations, the individual parts of the entropy are
			// calculated and summed.
			// i.e. if there is 0 of a class, then that part would calculate as NaN,
			// but this can be caught and
			// set to 0.
			ArrayList<Double> entropyParts = new ArrayList<Double>();
			for (Double d : classDistributions.keySet()) {
				thisPart = (double) classDistributions.get(d) / total;
				toAdd = -thisPart * Math.log10(thisPart) / Math.log10(2);
				if (Double.isNaN(toAdd))
					toAdd = 0;
				entropyParts.add(toAdd);
			}

			double entropy = 0;
			for (int i = 0; i < entropyParts.size(); i++) {
				entropy += entropyParts.get(i);
			}
			return entropy;
		}
		

		

		// comparison to determine order of shapelets in terms of info gain,
		// then separation gap, then shortness
		public int compareTo(Shapelet shapelet) {
			final int BEFORE = -1;
			final int EQUAL = 0;
			final int AFTER = 1;

			if (this.getGainRatio() != shapelet.getGainRatio()) {
				if (this.getGainRatio() > shapelet.getGainRatio()) {
					return BEFORE;
				} else {
					return AFTER;
				}
			} else {// if this.informationGain == shapelet.informationGain
					if (this.content.length != shapelet.getLength()) {
						if (this.content.length < shapelet.getLength()) {
						return BEFORE;
					} else {
						return AFTER;
					}
				} else {
					return EQUAL;
				}
			}

		}
	
	}