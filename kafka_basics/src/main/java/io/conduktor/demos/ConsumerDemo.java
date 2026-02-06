package io.conduktor.demos;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.errors.WakeupException;
import org.apache.kafka.common.serialization.StringDeserializer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.Arrays;
import java.util.Properties;

public class ConsumerDemo {

    private static final Logger log = LoggerFactory.getLogger(ConsumerDemo.class.getSimpleName());
    public static void main(String[] args) {
        log.info(" I am a Kafka consumer !");


        String groupId = "my-java-application";

        // create Producer properties
        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("key.deserializer",
                "org.apache.kafka.common.serialization.StringDeserializer");
        properties.setProperty("value.deserializer",
                "org.apache.kafka.common.serialization.StringDeserializer");

        // create consumer configs
        properties.setProperty("key.deserializer", StringDeserializer.class.getName());
        properties.setProperty("value.deserializer", StringDeserializer.class.getName());
        properties.setProperty("group.id", groupId);
        properties.setProperty("auto.offset.reset", "earliest");

        // create a consumer
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(properties);

        // get a reference to the main thread
        final Thread mainThread = Thread.currentThread();

        // adding the shutdown hook (Graceful shutdown)
        Runtime.getRuntime().addShutdownHook(new Thread() {
            public void run() {
                log.info("Detected a shutdown, lets exit by calling consumer.wakeup()...");
                consumer.wakeup();

                //  join the main thread to allow the execution of the code in the main thread
                // notice main thread to execute the shutdown
                try {
                    mainThread.join();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }

        });

        try {
            // subscribe to  a topic
            consumer.subscribe(Arrays.asList("topic"));

            // poll for data
            // long - running streaming service
            while (true) {

                log.info("Polling...");

                ConsumerRecords<String, String> records =
                        consumer.poll(Duration.ofMillis(1000));

                for (ConsumerRecord<String, String> record : records) {
                    log.info("Key = " + record.key() + ", Value = " + record.value());
                    log.info("Partition = " + record.partition() + ", Offset = " + record.offset());
                }
            }
        } catch (WakeupException e) {
            log.info("Consumer is starting to shut down");
        } catch (Exception e) {
            log.error("Unexpected exception in the consumer" , e);
        } finally{
            consumer.close();
            log.info("The consumer is gracefully shut down");
        }
    }

}
