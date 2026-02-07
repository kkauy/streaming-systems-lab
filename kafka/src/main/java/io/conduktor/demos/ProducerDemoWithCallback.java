package io.conduktor.demos;

import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.common.serialization.StringSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

public class ProducerDemoWithCallback {

    private static final Logger log = LoggerFactory.getLogger(ProducerDemoWithCallback.class.getSimpleName());
    public static void main(String[] args) {
        log.info(" I am a Kafka producer !");

        // create Producer properties
        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("key.serializer",
                "org.apache.kafka.common.serialization.StringSerializer");
        properties.setProperty("value.serializer",
                "org.apache.kafka.common.serialization.StringSerializer");

        // set producer properties
        properties.setProperty("key.serializer", StringSerializer.class.getName());
        properties.setProperty("value.serializer",StringSerializer.class.getName());

        // create the producer
        KafkaProducer<String,String> producer = new KafkaProducer<>(properties);


        // create a Producer Record
        ProducerRecord<String, String> producerRecord = new ProducerRecord<>("dem0_java", "hello Kafka");

        // send data


        // flush and close the producer

        producer.flush();

        producer.close();

    }

}
