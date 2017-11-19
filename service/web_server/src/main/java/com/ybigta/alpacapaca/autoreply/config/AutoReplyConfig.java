package com.ybigta.alpacapaca.autoreply.config;

import com.ybigta.alpacapaca.autoreply.service.ContentValidator;
import com.ybigta.alpacapaca.autoreply.service.validator.ContentValidatorImpl;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AutoReplyConfig {
    @Bean
    public ContentValidator payloadValidator() {
        return new ContentValidatorImpl();
    }
}
