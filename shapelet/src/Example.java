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
		 int difAggr = 1;
		 int numBins=13;
		 int numTrainInstances=20;
		 int numMax=641;
		    //Read and create train and test Instances
					Instances train=null, test=null, all=null;
					FileReader r;
					try{		
						r= new FileReader("Sony_TRAIN.arff"); 
						train = new Instances(r); 
						train.setClassIndex(0);
						r = new FileReader("Sony_TEST.arff"); 
					    test = new Instances(r);
						test.setClassIndex(0);
						r = new FileReader("Sony_ALL.arff"); 
					    all = new Instances(r);
						all.setClassIndex(0);
			                        
					}
					catch(Exception e)
					{
						System.out.println("Unable to load data. Exception thrown ="+e);
						System.exit(0);
					}
					
					
					

					
					ArrayList<Instances> datasets = new ArrayList<Instances>();
					datasets.add(train);

			//Assuming thar the training dataset is not discretized yet!
			train.setRelationName("0");
			test.setRelationName("0");
			all.setRelationName("0");
			System.out.println(all.numInstances());
			
			for(int x=0; x<difAggr;x++){
				//Discretize the dataset together!
				Instances dis = all;
				dis.setClassIndex(0);
				Discretize d = new Discretize();
				d.setInputFormat(dis);
				d.setBins(numBins);
				Instances output = Filter.useFilter(dis, d);
				output.setRelationName(""+numBins);
				save(output, "" + output.relationName() + "Aggr" + numBins);
				
				//Split the training and testing datasets already discretized
				int j=0;
				Instances train1 = new Instances(output, output.numInstances());
				for (j = 0; j < numTrainInstances; j++) {
					train1.add(output.instance(j));
				}
	
				train1.setRelationName(""+numBins);
				save(train1, "Train" + numBins);
				datasets.add(train1);
				
				Instances test1 = new Instances(output, output.numInstances());
				for(int i=numTrainInstances;i<numMax;i++){
					test1.add(output.instance(i));
				}
				
				test1.setRelationName(""+numBins);
				save(test1, "" + numBins);
				
				
				numBins +=1;
			}
			
		
			
			
			
			
			
			
				
								
					
			
					
			/*	//Discretize the training dataset into different aggregarions
				for(int i=0; i< difAggr; i++){
					Instances inputTrain = train;
					inputTrain.setClassIndex(0);
					Discretize d = new Discretize();
					d.setInputFormat(inputTrain);
					d.setBins(numBins);
					Instances outputTrain = Filter.useFilter(inputTrain, d);
					outputTrain.setRelationName(""+numBins);
					save(outputTrain, "" + outputTrain.relationName() + "Aggr" + numBins);
					datasets.add(outputTrain);
					numBins +=5;
				}
					
				
				numBins=5;
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
						
				datasets.add(train);*/
				//Create ShapeletTreeClassifier
				ShapeletTreeClassifierModified shapeletTree = new ShapeletTreeClassifierModified("log");
				shapeletTree.setShapeletMinMaxLength(5,70);
				shapeletTree.buildClassifier(datasets);
					
					/*Evaluation eval = new Evaluation(train);
					eval.crossValidateModel(shapeletTree, train, 2, new Random(1));
					System.out.println(eval.toSummaryString("\nResults\n======\n", false));*/
					
					//Evaluation
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(shapeletTree, test);
					System.out.println(eval.toSummaryString("\nResults\n======\n", false));
					System.out.println(eval.toMatrixString());
			
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