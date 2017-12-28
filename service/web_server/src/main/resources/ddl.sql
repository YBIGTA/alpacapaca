CREATE TABLE `alpacapaca_record` (
`id` bigint(20) NOT NULL AUTO_INCREMENT,
`input` varchar(255) DEFAULT NULL,
`output` varchar(255) DEFAULT NULL,
`request_time` bigint(20) DEFAULT NULL,
`user_key` varchar(255) DEFAULT NULL,
PRIMARY KEY (`id`)
) DEFAULT CHARSET=utf8;
