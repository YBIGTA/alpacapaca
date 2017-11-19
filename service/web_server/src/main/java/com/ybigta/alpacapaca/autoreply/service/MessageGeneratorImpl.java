package com.ybigta.alpacapaca.autoreply.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MessageGeneratorImpl implements MessageGenerator {
    private final ContentValidator contentValidator;

    @Autowired
    public MessageGeneratorImpl(final ContentValidator contentValidator) {
        this.contentValidator = contentValidator;
    }

    @Override
    public String generateMessage(String inputContent) {
        StringBuilder builder = new StringBuilder();

        if (contentValidator.validate(inputContent)) {
            builder.append(inputContent);
            builder.append(System.lineSeparator());
            builder.append("에 대한 응답입니다.");
        } else {
            builder.append("유효한 메세지가 아닙니다.");
        }

        return builder.toString();
    }
}
