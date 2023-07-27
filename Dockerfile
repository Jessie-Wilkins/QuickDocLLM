FROM rust:1.71-buster

COPY ./model/ /app/model

COPY ./src/ /app/src/

COPY Cargo.lock /app/Cargo.lock

COPY Cargo.toml /app/Cargo.toml

WORKDIR /app/

RUN apt-get update \
&& apt-get install -y clang

RUN rustup component add rustfmt

RUN RUST_BACKTRACE=1 cargo build

CMD ["cargo", "run"]