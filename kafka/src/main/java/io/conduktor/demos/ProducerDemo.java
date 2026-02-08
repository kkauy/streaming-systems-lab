package io.conduktor.demos;

import org.apache.kafka.clients.producer.Callback;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.clients.producer.RecordMetadata;
import org.apache.kafka.common.serialization.StringSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

public class ProducerDemo {

    private static final Logger log = LoggerFactory.getLogger(ProducerDemo.class.getSimpleName());
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


        properties.setProperty("batch.size", "400");

        // create the producer
        KafkaProducer<String,String> producer = new KafkaProducer<>(properties);

        // create a Producer Record
        for (int j =0; j < 10; j++) {

            for (int i = 0; i < 30;i++) {
                ProducerRecord<String, String> producerRecord =
                        new ProducerRecord<>("dem0_java", "hello Kafka" + i);

                // send data
                producer.send(producerRecord, new Callback() {
                    @Override
                    public void onCompletion(RecordMetadata recordMetadata, Exception e) {
                        // executes every time a record successfully send or an exception is thrown
                        if (e == null) {
                            // the record was successfully sent
                            log. info("Received new metadata \n"
                                    + "Topic:" + recordMetadata.topic() + "\n"
                                    + "Partition:" + recordMetadata.partition() + "\n"
                                    + "Offset:" + recordMetadata.offset() + "\n"
                                    + "Timestamp:" + recordMetadata.timestamp());
                        } else {
                            log.error("Error while producing", e);
                        }
                    }
                });

            }

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }



        // flush and close the producer

        producer.flush();

        producer.close();

    }

}
