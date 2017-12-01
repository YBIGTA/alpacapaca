package com.ybigta.alpacapaca.autoreply.service;

import com.ybigta.alpacapaca.autoreply.model.AlpacapacaMessage;
import com.ybigta.alpacapaca.autoreply.service.validator.ValidationResult;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Slf4j
@Service
public class MessageGeneratorImpl implements MessageGenerator {
    private static final String ENTER = "\n";
    private final ContentValidator contentValidator;
    @Value("${model-server-endpoint}")
    private String modelServerEndPoint;

    @Autowired
    public MessageGeneratorImpl(final ContentValidator contentValidator) {
        this.contentValidator = contentValidator;
    }

    @Override
    public String generateMessage(final String inputContent) {
        StringBuilder builder = new StringBuilder();
        ValidationResult validationResult = contentValidator.validate(inputContent);

        if (validationResult.isValid()) {
            ResponseEntity<AlpacapacaMessage> response = getAlpacapacaMessage(inputContent);
            if (response.getStatusCode() == HttpStatus.OK) {
                AlpacapacaMessage message = response.getBody();
                builder.append(toResultFormat(message));
            } else {
                log.info("Server Error Occurs");
                builder.append(ErrorMessages.SERVER_ERROR_MESSAGE);
            }
        } else {
            builder.append(validationResult.getMessage());
        }

        return builder.toString();
    }

    private String toResultFormat(final AlpacapacaMessage message) {
        StringBuilder builder = new StringBuilder();

        if (message.isSuccess() &&
                message.getResults() != null && message.getResults().size() == 3) {
            builder.append(String.join(ENTER, message.getResults()));
        } else {
            log.error("Result format error {}", message.getResults());
            builder.append(ErrorMessages.MESSAGE_GENERATING_FAILURE_MESSAGE);
        }

        return builder.toString();
    }

    private ResponseEntity<AlpacapacaMessage> getAlpacapacaMessage(final String inputContent) {
        RestTemplate restTemplate = new RestTemplate();
        return restTemplate.getForEntity(modelServerEndPoint + "/alpaca/" + inputContent, AlpacapacaMessage.class);
    }
}
