import java.io.FileReader;
import java.util.ArrayList;


import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.shapelet_tree.*;


public class Example {
	

	 public static void main(String[] args) throws Exception{
		 
		 
		    //Read and create train and test Instances
			Instances train=null, test=null, discretized=null;
			FileReader r;
			try{		
				r= new FileReader("Coffee_TRAIN.arff"); 
				train = new Instances(r); 
				train.setClassIndex(0);
				//r = new FileReader("Coffee_Discretized.arff"); 
			    //discretized = new Instances(r);
				//discretized.setClassIndex(0);
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
			datasets.add(test);

		
					
			//Create ShapeletTreeClassifier
			ShapeletTreeClassifierModified shapeletTree = new ShapeletTreeClassifierModified("log");
			shapeletTree.setShapeletMinMaxLength(10,286);
			shapeletTree.buildClassifier(datasets);
			
			/*Evaluation eval = new Evaluation(train);
			eval.crossValidateModel(shapeletTree, train, 5, new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));*/
			
			//Evaluation
	/*		Evaluation eval = new Evaluation(train);
			eval.evaluateModel(shapeletTree, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));*/
			
	
	 }
}
