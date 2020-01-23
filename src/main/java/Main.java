import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import scala.Tuple2;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;


public class Main {

    public static final String cf = "cf";
    public static final String cp = "cp";

    public static void main(String[] args) {
        String option = null;
        System.out.println("option choisi: " + args[0]);

        if ((args.length == 1)) {
            if (args[0].equals(cf)) {
                option = cf;
            } else {
                option = cp;
            }
        } else {
            System.out.println("Rentrer un parametre (cf/cp)");
            System.exit(1);
        }

        SparkConf conf = new SparkConf().setAppName("BigData").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);


        //(QUESTION - 7)
        SparkSession spark = SparkSession
                .builder()
                .appName("SparkSessionExample")
                .getOrCreate();


        //QUESTION-1
        String path = "EVC-TXT/" + option + "/*.txt";
        System.out.println(path);
        JavaRDD<String> doc = sc.textFile(path);


        //QUESTION-2
        //On separe les mots puis on les compte
        JavaPairRDD<String, Integer> listeMots = doc.flatMap(s -> Arrays.asList(s.split("[^a-zA-ZáàâäãéèêëíìîïóòôöõúùûüýÿÁÀÂÄÃÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÝ]")).iterator())
                .map(String::toLowerCase)
                .mapToPair(word -> new Tuple2<>(word, 1))
                .filter(word -> !word._1.equals(""));

        long totalMots = listeMots.count();

        //On rassemble les mots, les trie puis on les compte
        JavaPairRDD<String, Integer> listeMots2 = listeMots
                .reduceByKey((a, b) -> (a + b))
                .filter(word -> !word._1.equals(""))
                .mapToPair(s -> s.swap())
                .sortByKey(false)
                .mapToPair(s -> s.swap());

        long usageMots = listeMots2.count();

        //affichage
        System.out.println(listeMots2.collect());
        System.out.println("Total de mots dans les documents : " + totalMots);
        System.out.println("Différents mots utilisés : " + usageMots);

        //QUESTION-3
        List<String> stopwords = sc.textFile("french-stopwords.txt").collect();
        JavaPairRDD<String, Integer> listeMots3 = listeMots2.filter(word -> !stopwords.contains(word._1));

        //QUESTION-4
        //Les mots sont deja ordonnes, on peut donc prendre les 10 premiers
        System.out.println(listeMots3.take(10));


        //QUESTION-5 + 6
        //Path des documents
        File[] files = new File("EVC-TXT/"+option).listFiles();

        //Liste des transactions
        List<Row> data = new ArrayList<Row>();


        for (File f : files){
            JavaRDD<String> file = sc.textFile(f.getPath()).flatMap(s -> Arrays.asList(s.split("[^a-zA-ZáàâäãéèêëíìîïóòôöõúùûüýÿÁÀÂÄÃÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÝ]")).iterator())
                    .map(String::toLowerCase)
                    // On retire les mots correspondant aux stopwords
                    .filter(word -> !stopwords.contains(word))
                    .filter( word -> !word.equals(""));

            //On ajoute la rdd comme transaction
            data.add(RowFactory.create(file.collect().stream().distinct().collect(Collectors.toList())));
        }


        //QUESTION-7
        StructType schema = new StructType(new StructField[]{ new StructField(
                "items", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        Dataset<Row> itemsDF = spark.createDataFrame(data, schema);


        Scanner input = new Scanner(System.in);
        double minsup;
        double minconf;
        int topK;

        System.out.println("Veuillez saisir minsup : (default: 0,8)");
        minsup = input.nextDouble();
        System.out.println("Veuillez saisir minconf : (default: 0,8)");
        minconf = input.nextDouble();

        FPGrowthModel model = new FPGrowth()
                .setItemsCol("items")
                .setMinSupport(minsup)
                .setMinConfidence(minconf)
                .fit(itemsDF);

        System.out.println("Veuillez saisir top K : (default: 20)");
        topK = input.nextInt();

        // Display frequent itemsets.
        model.freqItemsets().show(topK,false);

        model.associationRules().show(topK, false);

        // Synthese presentant les consequences possibles comme predictions
        model.transform(itemsDF).show(topK,false);

    }
}
