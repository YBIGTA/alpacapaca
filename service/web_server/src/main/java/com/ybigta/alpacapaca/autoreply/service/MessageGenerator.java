package com.ybigta.alpacapaca.autoreply.service;

public interface MessageGenerator {
    /**
     * 사용자가 입력한 내용을 바탕으로 3행시를 생성합니다.
     * 입력한 내용을 Validator 를 이용해 검증하고 적합하다면 3행시를 생성해 제공합니다.
     *
     * @param inputContent 사용자가 채팅창을 통해 입력한 내용
     * @return inputContent 가 적합하다면 3행시가 그렇지 않다면 그에 알맞은 문구가 반환됩니다.
     */
    String generateMessage(final String inputContent);
}
