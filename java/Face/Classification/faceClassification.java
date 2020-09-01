package Face.Classification;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
//import Core.Math.EuclideanDistance;

import java.io.File;

public class faceClassification {
    private static FaceNetToVector _createInputVector;
//    private FaceNetToVector _createInputVector;

    public static void main (String [] args){
        _createInputVector =new FaceNetToVector();
        Euclidean _distance = new Euclidean();
        File file1 = new File("C:/Users/TRAINING/Desktop/trump.jpg");
        File file2 = new File("C:/Users/TRAINING/Desktop/trump2.jpg");
        File file3 = new File("C:/Users/TRAINING/Desktop/najib.jpg");
        System.out.println("Recognize: "+ file1 + " " + file2 + " " + file3);

        // Transform into embeddings
        double[] faceFeatureArray1 = _createInputVector.runTask(file1);
        double[] faceFeatureArray2 = _createInputVector.runTask(file2);
        double[] faceFeatureArray3 = _createInputVector.runTask(file3);
        INDArray array1 = Nd4j.create(faceFeatureArray1);
        INDArray array2 = Nd4j.create(faceFeatureArray2);
        INDArray array3 = Nd4j.create(faceFeatureArray3);
//        System.out.println(array1);
//        System.out.println(array2);
        double distance1 = _distance.run(array1,array2);
        double distance2 = _distance.run(array1,array3);
        double distance3 = _distance.run(array2,array3);
        System.out.println("Trump to Trump2: " + distance1);
        System.out.println("Trump to najib: " + distance2);
        System.out.println("Trump2 to najib: " + distance3);
//        double minimalDistance = Double.MAX_VALUE;
//        String result = "";
//        for(
//                PersonModel personModel :_trainList)
//
//        {
//            INDArray array2 = Nd4j.create(personModel.get_faceFeatureArray());
//            double distance = _distance.run(array1, array2);
//            if (distance < minimalDistance) {
//                minimalDistance = distance;
//                result = personModel.get_personName();
//            }
//        }
    }

}
