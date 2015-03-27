package weka.core.shapelet;



import java.io.FileWriter;
import java.util.ArrayList;
import java.util.TreeMap;


//shapelet Class
public class Shapelet implements Comparable<Shapelet> {
 
    //shapelet Class

public double[] content; 
public int seriesId;
public int startPos;
public double splitThreshold;
public double splitInfo;
public double informationGain;
public double separationGap;
private double gainRatio;
public int granularity;
public int numBins;


//Constructors
public Shapelet(double[] content, int seriesId, int startPos, int granularity, int numBins) {
	this.setContent(content);
	this.setSeriesId(seriesId);
	this.setStartPos(startPos);
	this.setGranularity(granularity);
	this.setnumBins(numBins);

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

private void setGranularity(int granularity) {
	this.granularity = granularity;
}

public int getGranularity(){
	return this.granularity;
}

public double getSplitInfo(){
	return this.splitInfo;
}

public void setSplitInfo(double splitInfo){
	this.splitInfo = splitInfo;
}

public void setnumBins(int numBins) {
	this.numBins=numBins;
	
}

public int getNumBins(){
	return this.numBins;
}

 public double getGap(){
        return this.separationGap;
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
	double threshold = -1;
	double Infogain = 0;
	double splitInfoValue = -1.0;
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

			
			splitInfoValue = splitInfo(orderline);
			
			gainRatio = Infogain/splitInfoValue;
			
			
			
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
		this.setSplitThreshold(threshold);
		this.setSplitInfo(splitInfoValue);
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


 public void calcInfoGainAndThreshold(ArrayList<OrderLineObj> orderline, TreeMap<Double, Integer> classDistribution){
        // for each split point, starting between 0 and 1, ending between end-1 and end
        // addition: track the last threshold that was used, don't bother if it's the same as the last one
        double lastDist = orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
        double thisDist = -1;

        double bsfGain = -1;
        double threshold = -1;

        for(int i = 1; i < orderline.size(); i++){
            thisDist = orderline.get(i).getDistance();
            if(i==1 || thisDist != lastDist){ // check that threshold has moved(no point in sampling identical thresholds)- special case - if 0 and 1 are the same dist

                // count class instances below and above threshold
                TreeMap<Double, Integer> lessClasses = new TreeMap<Double, Integer>();
                TreeMap<Double, Integer> greaterClasses = new TreeMap<Double, Integer>();

                for(double j : classDistribution.keySet()){
                    lessClasses.put(j, 0);
                    greaterClasses.put(j, 0);
                }

                int sumOfLessClasses = 0;
                int sumOfGreaterClasses = 0;

                //visit those below threshold
                for(int j = 0; j < i; j++){
                    double thisClassVal = orderline.get(j).getClassVal();
                    int storedTotal = lessClasses.get(thisClassVal);
                    storedTotal++;
                    lessClasses.put(thisClassVal, storedTotal);
                    sumOfLessClasses++;
                }

                //visit those above threshold
                for(int j = i; j < orderline.size(); j++){
                    double thisClassVal = orderline.get(j).getClassVal();
                    int storedTotal = greaterClasses.get(thisClassVal);
                    storedTotal++;
                    greaterClasses.put(thisClassVal, storedTotal);
                    sumOfGreaterClasses++;
                }

                int sumOfAllClasses = sumOfLessClasses + sumOfGreaterClasses;

                double parentEntropy = entropy(classDistribution);

                // calculate the info gain below the threshold
                double lessFrac =(double) sumOfLessClasses / sumOfAllClasses;
                double entropyLess = entropy(lessClasses);
                // calculate the info gain above the threshold
                double greaterFrac =(double) sumOfGreaterClasses / sumOfAllClasses;
                double entropyGreater = entropy(greaterClasses);

                double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;

                if(gain > bsfGain){
                    bsfGain = gain;
                    threshold =(thisDist - lastDist) / 2 + lastDist;
                }
            }
            lastDist = thisDist;
        }
        if(bsfGain >= 0){
            this.informationGain = bsfGain;
            this.splitThreshold = threshold;
            this.separationGap = calculateSeparationGap(orderline, threshold);
        }
    }
 
 
 
 
 
 private double calculateSeparationGap(ArrayList<OrderLineObj> orderline, double distanceThreshold){

        double sumLeft = 0;
        double leftSize = 0;
        double sumRight = 0;
        double rightSize = 0;

        for(int i = 0; i < orderline.size(); i++){
            if(orderline.get(i).getDistance() < distanceThreshold){
                sumLeft += orderline.get(i).getDistance();
                leftSize++;
            } else{
                sumRight += orderline.get(i).getDistance();
                rightSize++;
            }
        }

        double thisSeparationGap = 1 / rightSize * sumRight - 1 / leftSize * sumLeft; //!!!! they don't divide by 1 in orderLine::minGap(int j)

        if(rightSize == 0 || leftSize == 0){
            return -1; // obviously there was no seperation, which is likely to be very rare but i still caused it!
        }                //e.g if all data starts with 0, first shapelet length =1, there will be no seperation as all time series are same dist
        // equally true if all data contains the shapelet candidate, which is a more realistic example

        return thisSeparationGap;
    }

 

// comparison to determine order of shapelets in terms of gain ration then shortness
 // comparison 1: to determine order of shapelets in terms of info gain, then separation gap, then shortness
public int compareTo(Shapelet shapelet) {
    final int BEFORE = -1;
    final int EQUAL = 0;
    final int AFTER = 1;

    

    if(this.informationGain != shapelet.getInformationGain()){
        if(this.informationGain > shapelet.getInformationGain()){
            return BEFORE;
        }else{
            return AFTER;
        }
    } else{// if this.informationGain == shapelet.informationGain
       /* if(this.separationGap != shapelet.getGap()){
            if(this.separationGap > shapelet.getGap()){
                return BEFORE;
            }else{
                return AFTER;
            }
        } else if(this.content.length != shapelet.getLength()){
            if(this.content.length < shapelet.getLength()){
                return BEFORE;
            }else{
                return AFTER;
            }
        } else{*/
            return EQUAL;
        }
    }



}
