// Java implementation of drug design exemplar for Hadoop

package edu.stolaf.cs;

import java.io.IOException;
import java.util.*;
import java.lang.Math;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class DDHadoop {

  static final String protein = 
    new String("the cat in the hat wore the hat to the cat hat party");

  public static void main(String[] args) throws Exception {
    JobConf conf = new JobConf(DDHadoop.class);
    conf.setJobName("DDHadoop");
    
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(Text.class);

    conf.setMapperClass(Map.class);
    conf.setCombinerClass(Reduce.class);
    conf.setReducerClass(Reduce.class);

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);

    FileInputFormat.setInputPaths(conf, new Path(args[0]));
    FileOutputFormat.setOutputPath(conf, new Path(args[1]));
    
    JobClient.runJob(conf);
  }

  static int score(String str1, String str2) {
    if (str1.equals("") || str2.equals(""))
      return 0;
    // both argument strings non-empty
    if (str1.charAt(0) == str2.charAt(0))
      return 1 + score(str1.substring(1), str2.substring(1));
    else // first characters do not match
      return Math.max(score(str1, str2.substring(1)), 
		      score(str1.substring(1), str2));
  }
  
  public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {

    public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
      String ligand = value.toString();
      output.collect(new IntWritable(score(ligand, protein)), value);
    }
  }

  public static class Reduce extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
    public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
      String result = new String("");
      while (values.hasNext()) {
        result += values.next().toString() + " ";
      }

      output.collect(key, new Text(result));
    }
  }

}

