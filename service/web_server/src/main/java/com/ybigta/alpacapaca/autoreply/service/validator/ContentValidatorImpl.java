package com.ybigta.alpacapaca.autoreply.service.validator;

import com.ybigta.alpacapaca.autoreply.service.ContentValidator;

import java.util.regex.Pattern;

public class ContentValidatorImpl implements ContentValidator {
    private static final int PROPER_INPUT_STRING_LENGTH = 3;
    private static final Pattern KOREAN_LANGUAGE_PATTERN = Pattern.compile("^[ㄱ-ㅎ가-힣]*$");

    @Override
    public boolean validate(String inputContent) {
        return inputContent != null
                && isProperLength(inputContent)
                && isKorean(inputContent);
    }

    private boolean isProperLength(String inputContent) {
        return inputContent.length() == PROPER_INPUT_STRING_LENGTH;
    }

    private boolean isKorean(String inputContent) {
        return KOREAN_LANGUAGE_PATTERN.matcher(inputContent).matches();
    }
}
