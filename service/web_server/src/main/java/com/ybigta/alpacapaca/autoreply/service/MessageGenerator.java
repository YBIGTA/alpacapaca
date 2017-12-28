package com.ybigta.alpacapaca.autoreply.service;

public interface MessageGenerator {
    /**
     * 사용자가 입력한 내용을 바탕으로 3행시를 생성합니다.
     * 입력한 내용을 Validator 를 이용해 검증하고 적합하다면 3행시를 생성해 제공합니다.
     *
     * @param inputContent 사용자가 채팅창을 통해 입력한 내용
     * @return Message generation 의 성공여부와 메시지를 담은 객체가 반환됩니다.
     */
    MessageGenerationResult generateMessage(final String inputContent);
}
