package io.conduktor.demos;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
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
        properties.setProperty("value.deserializer",StringDeserializer.class.getName());
        properties.setProperty("group.id", groupId);

        properties.setProperty("auto, offset.reset", "none/earliest/latest");

        // create a consumer
        KafkaConsumer<String, String> consumer = new KafkaConsumer<String, String>(properties);


        // subscribe to a topic
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
    }

}
