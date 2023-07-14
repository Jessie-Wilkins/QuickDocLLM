use llm_chain::executor;
use llm_chain::options;
use llm_chain::options::{ModelRef, Options};
use std::error::Error;

use llm_chain::{prompt::Data, traits::Executor};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {

    let model_type = "llama";
    let model_path = "model/orca-mini-3b.ggmlv3.q4_1.bin";
    let prompt = "Who was the president in 1985?";

    let exec = executor!(
        llama,
        options!(
            Model: ModelRef::from_path(model_path),
            ModelType: model_type.to_string()
        )
    )?;
    let res = prompt!(prompt).run(&parameters!(), &exec).await?;

    println!("{}", res);
    Ok(())
}