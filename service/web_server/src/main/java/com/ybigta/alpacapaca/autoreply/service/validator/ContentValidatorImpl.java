package com.ybigta.alpacapaca.autoreply.service.validator;

import com.ybigta.alpacapaca.autoreply.service.ContentValidator;
import com.ybigta.alpacapaca.autoreply.service.ErrorMessages;
import lombok.extern.slf4j.Slf4j;

import java.util.regex.Pattern;

@Slf4j
public class ContentValidatorImpl implements ContentValidator {
    private static final int PROPER_INPUT_STRING_LENGTH = 3;
    private static final Pattern KOREAN_LANGUAGE_PATTERN = Pattern.compile("^[가-힣]*$");

    @Override
    public ValidationResult validate(final String inputContent) {
        ValidationResult validationResult = new ValidationResult();

        if (inputContent == null) {
            log.info("validation fail, inputContent: {}, reason: {}",
                    inputContent, ErrorMessages.INPUT_MESSAGE_IS_NULL);
            validationResult.setValid(false);
            validationResult.setMessage(ErrorMessages.INPUT_MESSAGE_IS_NULL);
            return validationResult;
        }

        if (!isProperLength(inputContent)) {
            log.info("validation fail, inputContent: {}, reason: {}",
                    inputContent, ErrorMessages.LENGTH_IS_NOT_VALID_MESSAGE);
            validationResult.setValid(false);
            validationResult.setMessage(ErrorMessages.LENGTH_IS_NOT_VALID_MESSAGE);
            return validationResult;
        }

        if (!isKorean(inputContent)) {
            log.info("validation fail, inputContent: {}, reason: {}",
                    inputContent, ErrorMessages.INPUT_CONTENT_IS_NOT_KOREAN_MESSAGE);
            validationResult.setValid(false);
            validationResult.setMessage(ErrorMessages.INPUT_CONTENT_IS_NOT_KOREAN_MESSAGE);
            return validationResult;
        }

        validationResult.setValid(true);
        return validationResult;
    }

    private boolean isProperLength(final String inputContent) {
        return inputContent.length() == PROPER_INPUT_STRING_LENGTH;
    }

    private boolean isKorean(final String inputContent) {
        return KOREAN_LANGUAGE_PATTERN.matcher(inputContent).matches();
    }
}
