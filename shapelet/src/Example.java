import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.shapelet_tree.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class Example {
	

	 public static void main(String[] args) throws Exception{
		 int difAggr = 2;
			int numBins=10;
		    //Read and create train and test Instances
					Instances train=null, test=null, discretized=null;
					FileReader r;
					try{		
						r= new FileReader("Coffee_TRAIN.arff"); 
						train = new Instances(r); 
						train.setClassIndex(0);
						r = new FileReader("Coffee_TEST.arff"); 
					    test = new Instances(r);
						test.setClassIndex(0);
			                        
					}
					catch(Exception e)
					{
						System.out.println("Unable to load data. Exception thrown ="+e);
						System.exit(0);
					}
					
					
					
				
					
					
					ArrayList<Instances> datasets = new ArrayList<Instances>();
					

			//Assuming thar the training dataset is not discretized yet!
			train.setRelationName("0");
			test.setRelationName("0");
					
				//Discretize the training dataset into different aggregarions
				for(int i=0; i< difAggr; i++){
					Instances inputTrain = train;
					Discretize d = new Discretize();
					d.setInputFormat(inputTrain);
					d.setBins(numBins);
					Instances outputTrain = Filter.useFilter(inputTrain, d);
					outputTrain.setRelationName(""+numBins);
					save(outputTrain, "" + outputTrain.relationName() + "Aggr" + numBins);
					datasets.add(outputTrain);
					numBins +=5;
				}
				

				datasets.add(train);
				
				numBins=10;
				// TO DO: Use the same intervals as in the trainingset. DOn't know how to do it yet.
				//Discretize the testing dataset with the same aggregarions
				for(int i=0; i< difAggr; i++){
					Instances inputTest = test;
					Discretize d = new Discretize();
					d.setInputFormat(inputTest);
					d.setBins(numBins);
					Instances outputTrain = Filter.useFilter(inputTest, d);
					outputTrain.setRelationName(""+numBins);
					save(outputTrain, "" + numBins);
					numBins +=5;
				}
						
				
				//Create ShapeletTreeClassifier
				ShapeletTreeClassifierModified shapeletTree = new ShapeletTreeClassifierModified("log");
				shapeletTree.setShapeletMinMaxLength(285,285);
				shapeletTree.buildClassifier(datasets);
					
					/*Evaluation eval = new Evaluation(train);
					eval.crossValidateModel(shapeletTree, train, 2, new Random(1));
					System.out.println(eval.toSummaryString("\nResults\n======\n", false));*/
					
					//Evaluation
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(shapeletTree, test);
					System.out.println(eval.toSummaryString("\nResults\n======\n", false));
					
			
			 }
	 
		/**
	   * saves the data to the specified file
	   *
	   * @param data        the data to save to a file
	   * @param filename    the file to save the data to
	   * @throws Exception  if something goes wrong
	   */
	  protected static void save(Instances data, String filename) throws Exception {
	    BufferedWriter  writer;

	    writer = new BufferedWriter(new FileWriter(filename+".arff"));
	    writer.write(data.toString());
	    writer.newLine();
	    writer.flush();
	    writer.close();
	  }
		}