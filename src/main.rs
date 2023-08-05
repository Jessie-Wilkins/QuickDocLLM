use std::io::Write;
use llm::ModelArchitecture::Llama;
use llm::{Model, 
    OutputRequest, ModelParameters};

use hora::core::ann_index::ANNIndex;
use hora::index::hnsw_idx::HNSWIndex;

use pdf_extract::extract_text;

use itertools::Itertools;

use std::path::Path;

pub struct EmbeddingService {
    model: Box<dyn Model>,
    index: HNSWIndex<f32, usize>,
    documents: Vec<String>,
}

impl EmbeddingService {
    pub fn new() -> Self {
        let dimension = 4096;
        let parameters = ModelParameters {            
            prefer_mmap: true,
            context_size: 2048,
            lora_adapters: None,
            use_gpu: true,
            gpu_layers: None,
        };
        let model = llm::load_dynamic(
            Some(Llama),
            // path to GGML file
            std::path::Path::new("model/llama-2-7b-chat.ggmlv3.q2_K.bin"),
            llm::TokenizerSource::Embedded,
            // llm::ModelParameters
            parameters,
            // load progress callback
            llm::load_progress_callback_stdout
        )
        .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

        let index = HNSWIndex::<f32, usize>::new(
            dimension,
            &hora::index::hnsw_params::HNSWParams::<f32>::default(),
        );

        let documents = Vec::new();

        Self { model, index, documents }
    }

    pub fn infer(&self, query: &str) {
        let mut session = self.model.start_session(Default::default());
        let res = session.infer::<std::convert::Infallible>(
            // model to use for text generation
            self.model.as_ref(),
            // randomness provider
            &mut rand::thread_rng(),
            // the prompt to use for text generation, as well as other
            // inference parameters
            &llm::InferenceRequest {
                prompt: query.into(),
                parameters: &llm::InferenceParameters::default(),
                play_back_previous_tokens: false,
                maximum_token_count: Some(50),
            },
            // llm::OutputRequest
            &mut Default::default(),
            // output callback
            |r| match r {
                llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                    print!("{t}");
                    std::io::stdout().flush().unwrap();
                    Ok(llm::InferenceFeedback::Continue)
                }
                _ => Ok(llm::InferenceFeedback::Continue),
            }
        );
        match res {
            Ok(result) => println!("\n\nInference stats:\n{result}"),
            Err(err) => println!("\n{err}"),
        }
    }

    pub fn get_embeddings(&self, query: &str) -> Vec<f32> {
        let mut session = self.model.start_session(Default::default());
        let mut output_request = OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };
        let vocab = self.model.tokenizer();
        let beginning_of_sentence = true;
        let query_token_ids = vocab
            .tokenize(query, beginning_of_sentence)
            .unwrap()
            .iter()
            .map(|(_, tok)| *tok)
            .collect::<Vec<_>>();
        self.model.evaluate(&mut session, &query_token_ids, &mut output_request);
        output_request.embeddings.unwrap()
    }

    pub fn add_to_index(&mut self, id: usize, vector: &Vec<f32>) {
        self.index.add(vector, id).unwrap();
    }

    pub fn build_index(&mut self, path: &str) {
        self.index.build(hora::core::metrics::Metric::DotProduct).unwrap();   
        self.index.dump(path);
    }

    pub fn load_index(&mut self, path: &str) {
        self.index.load(path);
    }

    pub fn query(&self, vector: &Vec<f32>, num_results: usize) -> Vec<usize> {
        self.index.search(vector, num_results)
    }

    pub fn add_document(&mut self, document: String) {
        let embeddings = self.get_embeddings(&document);
        self.add_to_index(self.documents.len(), &embeddings);
        self.documents.push(document);
    }

    pub fn ask_question(&self, question: &str) -> Vec<&String> {
        let question_embeddings = self.get_embeddings(question);
        let results = self.query(&question_embeddings, 10);

        results.iter().map(|&id| &self.documents[id]).collect()
    }
}

pub struct DocParser {}

impl DocParser {
    pub fn extract_from_pdf(file_path: String) -> Vec<String> {
        Self::split_string_to_chunks(100, extract_text(file_path).unwrap())
    }

    fn split_string_to_chunks(size_of_char_chunks: usize, text: String) -> Vec<String> {
        let mut sections = Vec::new();
    
        for chunk in &text.chars().chunks(size_of_char_chunks) {
            sections.push(String::from_iter(chunk));
            
        }

        sections

    }
}


fn main() {
// load a GGML model from disk


    let mut embedding_service = EmbeddingService::new();

    let mut embedding_count = 1;

    let path = "index/index.idx";

    if Path::new(&path).exists() {
        embedding_service.load_index(path);
    } 
    else {
        let extracted_doc = DocParser::extract_from_pdf("/Users/jessiewilkins/Documents/Wilkins Edited_Resume_2019v4_Software_Engineer.pdf".to_string());

        for chunk in &extracted_doc {
            println!("Generating Embedding {} of {}...", embedding_count, extracted_doc.len());
            embedding_service.add_document(chunk.to_string());
            embedding_count += 1;
        }
    
        embedding_service.build_index(&path);
    }

    let question = "What was Jessie Wilkins' first job?";

    let answers = embedding_service.ask_question(question);

    let answers: Vec<&str> = answers.iter().map(|s| s.as_str()).collect();

    let doc_query_attach = answers.join("\n");

    let res = embedding_service.infer(&format!("<human>:You will give a brief answer to a question using the following text between the <doc> tags:\n\n<doc>\n{doc_query_attach}\n<doc>\n
Here is the question: {question}\n<bot>:"));

}
