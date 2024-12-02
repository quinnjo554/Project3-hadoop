import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputPath;
import org.apache.hadoop.mapreduce.lib.output.FileOutputPath;

public class ChromosomeInteractionCounter {
    // Base pair lengths for each chromosome (matching Table 1)
    private static final int[] CHROMOSOME_LENGTHS = {
        248_956_422, 242_193_529, 198_295_559, 190_214_555, 181_538_259,
        170_805_979, 159_345_973, 145_138_636, 138_394_717, 133_797_422,
        135_086_622, 133_275_309, 114_364_328, 107_043_718, 101_991_189,
        90_338_345, 83_257_441, 80_373_285, 58_617_616, 64_444_167,
        46_709_983, 50_818_468, 156_040_895 // Last entry is X chromosome
    };

    // Bin size constant
    private static final int BIN_SIZE = 100_000;

    // Mapper class to process input interactions
    public static class InteractionMapper 
        extends Mapper<Object, Text, Text, IntWritable> {
        
        private Text binPair = new Text();
        private IntWritable one = new IntWritable(1);

        @Override
        public void map(Object key, Text value, Context context
                        ) throws IOException, InterruptedException {
            String line = value.toString().trim();
            StringTokenizer tokenizer = new StringTokenizer(line);
            
            // Ensure valid number of tokens
            if (tokenizer.countTokens() != 4) {
                return;
            }

            try {
                // Parse chromosome and position information
                int chrom1 = Integer.parseInt(tokenizer.nextToken());
                int pos1 = Integer.parseInt(tokenizer.nextToken());
                int chrom2 = Integer.parseInt(tokenizer.nextToken());
                int pos2 = Integer.parseInt(tokenizer.nextToken());

                // Validate chromosomes and positions
                if (chrom1 < 1 || chrom1 > 23 || 
                    chrom2 < 1 || chrom2 > 23 ||
                    pos1 > CHROMOSOME_LENGTHS[chrom1-1] ||
                    pos2 > CHROMOSOME_LENGTHS[chrom2-1]) {
                    return;
                }

                // Calculate global bin numbers
                int bin1 = calculateGlobalBinNumber(chrom1, pos1);
                int bin2 = calculateGlobalBinNumber(chrom2, pos2);

                // Ensure first bin is always less than second bin
                if (bin1 > bin2) {
                    int temp = bin1;
                    bin1 = bin2;
                    bin2 = temp;
                }

                // Create bin pair key
                binPair.set("(" + bin1 + ", " + bin2 + ")");
                context.write(binPair, one);

            } catch (NumberFormatException e) {
                // Skip invalid lines
                return;
            }
        }

        // Calculate global bin number across chromosomes
        private int calculateGlobalBinNumber(int chromosome, int position) {
            int binOffset = 0;
            
            // Calculate cumulative bins for previous chromosomes
            for (int i = 0; i < chromosome - 1; i++) {
                binOffset += (int) Math.ceil(CHROMOSOME_LENGTHS[i] / (double) BIN_SIZE);
            }

            // Calculate bin number for current chromosome
            return binOffset + (int) Math.ceil(position / (double) BIN_SIZE);
        }
    }

    // Reducer class to aggregate interaction counts
    public static class InteractionReducer 
        extends Reducer<Text, IntWritable, Text, IntWritable> {
        
        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, 
                           Context context
                           ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Chromosome Interaction Counter");
        job.setJarByClass(ChromosomeInteractionCounter.class);
        
        job.setMapperClass(InteractionMapper.class);
        job.setCombinerClass(InteractionReducer.class);
        job.setReducerClass(InteractionReducer.class);
        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        FileInputPath.addInputPath(job, new Path(args[0]));
        FileOutputPath.setOutputPath(job, new Path(args[1]));
        
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}