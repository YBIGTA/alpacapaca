package com.ybigta.alpacapaca.autoreply.service;

import com.ybigta.alpacapaca.autoreply.service.validator.ValidationResult;

public interface ContentValidator {
    /**
     * inputContent 가 정해진 규칙에 맞는지 여부를 판단합니다.
     *
     * @param inputContent 사용자가 채팅창을 통해 입력한 내용
     * @return 규칙에 맞는다면 true 를 담은 ValidationResult 를 반환합니다.
     * 그렇지 않다면 false 와 valid 하지 않은 이유를 담은 ValidationResult 를 반환합니다.
     */
    ValidationResult validate(final String inputContent);
}
