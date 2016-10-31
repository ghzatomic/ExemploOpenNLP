package br.com.caiopaulucci;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;

import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;

public class NLP{

	private static final String CORPUS = "<START:Pessoa> Marcelo <END> foi trabalhar em <START:Cidade> Piratininga <END>";
	private static final String TEXT = "Juliana trabalha em Ubatuba";
	private static final String MODEL_TOKENIZER = "data/pt-token.bin";
	private static final Integer ITERATION = 100;
	private static final Integer CUTOFF = 0;

	public static void main( String[ ] args ) throws IOException {
		InputStream model = trainer( new InputStreamFactory(){
			public InputStream createInputStream() throws IOException {
				return new ByteArrayInputStream( CORPUS.getBytes() );
			}
		} );

		classifier( model );
	}

	private static void classifier( InputStream stream ) throws InvalidFormatException, FileNotFoundException, IOException {
		TokenizerModel tokenModel = new TokenizerModel( new FileInputStream( new File( MODEL_TOKENIZER ) ) );
		Tokenizer token = new TokenizerME( tokenModel );
		Span indexes[] = token.tokenizePos( TEXT );

		TokenNameFinderModel model = new TokenNameFinderModel( stream );
		NameFinderME finder = new NameFinderME( model );

		String tokens[] = token.tokenize( TEXT );
		Span spans[] = finder.find( tokens );
		double probs[] = finder.probs( spans );

		for ( int i = 0; i < spans.length; i++ ) {
			Span span = spans[ i ];
			String textoclassificado = TEXT.substring( indexes[ span.getStart() ].getStart(), indexes[ span.getEnd() - 1 ].getEnd() );
			System.out.println( textoclassificado + " : " + probs[ i ] + " - " + span.getType() );
		}
	}

	private static InputStream trainer( InputStreamFactory inputStreamFactory ) throws IOException {
		ObjectStream< String > fileStream = new PlainTextByLineStream( inputStreamFactory, Charset.forName( "UTF-8" ) );
		ObjectStream< NameSample > sampleStream = new NameSampleDataStream( fileStream );
		TokenNameFinderFactory tokenFactory = new TokenNameFinderFactory();
		TrainingParameters params = new TrainingParameters();
		params.put( TrainingParameters.ITERATIONS_PARAM, Integer.toString( ITERATION ) );
		params.put( TrainingParameters.CUTOFF_PARAM, Integer.toString( CUTOFF ) );

		ByteArrayOutputStream output = new ByteArrayOutputStream();
		TokenNameFinderModel model = NameFinderME.train( "pt-br", "model.ser", sampleStream, params, tokenFactory );
		sampleStream.close();

		model.serialize( output );
		output.close();
		return new ByteArrayInputStream( output.toByteArray() );
	}

}