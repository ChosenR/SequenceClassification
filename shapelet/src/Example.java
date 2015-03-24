import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instances;
import weka.core.shapelet.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.shapelet_tree.*;
import weka.core.*;

public class Example {
	

	 public static void main(String[] args) throws Exception{
		 
		    //Read and create train and test Instances
					Instances train=null, test=null, discretized=null;
					FileReader r;
					try{		
						r= new FileReader("Coffee_T1.arff"); 
						train = new Instances(r); 
						train.setClassIndex(0);
						r = new FileReader("Coffee_D10.arff"); 
					    discretized = new Instances(r);
						discretized.setClassIndex(0);
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
					datasets.add(train);
					datasets.add(discretized);
			

						
					//Create ShapeletTreeClassifier
					ShapeletTreeClassifierModified shapeletTree = new ShapeletTreeClassifierModified("log");
					shapeletTree.setShapeletMinMaxLength(5,5);
					shapeletTree.buildClassifier(datasets);
					
					/*Evaluation eval = new Evaluation(train);
					eval.crossValidateModel(shapeletTree, train, 5, new Random(1));
					System.out.println(eval.toSummaryString("\nResults\n======\n", false));*/
					
					//Evaluation
					Evaluation eval = new Evaluation(train);
					eval.evaluateModel(shapeletTree, test);
					System.out.println(eval.toSummaryString("\nResults\n======\n", false));
					
			
			 }
		}