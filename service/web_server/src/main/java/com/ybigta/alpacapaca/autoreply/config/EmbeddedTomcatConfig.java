package com.ybigta.alpacapaca.autoreply.config;

import lombok.extern.slf4j.Slf4j;
import org.apache.catalina.valves.AccessLogValve;
import org.springframework.boot.context.embedded.ConfigurableEmbeddedServletContainer;
import org.springframework.boot.context.embedded.EmbeddedServletContainerCustomizer;
import org.springframework.boot.context.embedded.tomcat.TomcatEmbeddedServletContainerFactory;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurerAdapter;

@Slf4j
@Configuration
public class EmbeddedTomcatConfig extends WebMvcConfigurerAdapter implements EmbeddedServletContainerCustomizer {

    @Override
    public void customize(ConfigurableEmbeddedServletContainer container) {
        if (container instanceof TomcatEmbeddedServletContainerFactory) {
            TomcatEmbeddedServletContainerFactory factory = (TomcatEmbeddedServletContainerFactory) container;

            AccessLogValve accessLogValve = new AccessLogValve();
            accessLogValve.setDirectory("/ybigta/logs/bootapp/access");
            accessLogValve.setPattern("combined");
            accessLogValve.setSuffix(".log");

            factory.addContextValves(accessLogValve);
        } else {
            log.error("WARNING! this customizer does not support your configured container");
        }
    }
}