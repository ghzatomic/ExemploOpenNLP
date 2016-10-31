package br.com.caiopaulucci;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;

import opennlp.tools.ml.maxent.GIS;
import opennlp.tools.ml.maxent.io.GISModelReader;
import opennlp.tools.ml.maxent.io.PlainTextGISModelWriter;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.DataReader;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.PlainTextFileDataReader;
import opennlp.tools.ml.model.TwoPassDataIndexer;
import opennlp.tools.util.ObjectStream;

public class ClassificadorCondominio{
	public static final Integer ITERATION = 100;
	public static final Integer CUTOFF = 0;
	
	public static final String TRAINER_SET =
		"INADIMPLENTE=Nao CASA_PROPRIA=Sim ESTADO_CIVIL=Solteiro RENDA_ANUAL=125".trim() + System.lineSeparator() +
		"INADIMPLENTE=Nao CASA_PROPRIA=Nao ESTADO_CIVIL=Casado RENDA_ANUAL=100 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Nao CASA_PROPRIA=Nao ESTADO_CIVIL=Solteiro RENDA_ANUAL=70 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Nao CASA_PROPRIA=Sim ESTADO_CIVIL=Casado RENDA_ANUAL=120 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Sim CASA_PROPRIA=Nao ESTADO_CIVIL=Divorciado RENDA_ANUAL=95 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Nao CASA_PROPRIA=Nao ESTADO_CIVIL=Casado RENDA_ANUAL=60 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Nao CASA_PROPRIA=Sim ESTADO_CIVIL=Divorciado RENDA_ANUAL=220 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Sim CASA_PROPRIA=Nao ESTADO_CIVIL=Solteiro RENDA_ANUAL=85 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Nao CASA_PROPRIA=Nao ESTADO_CIVIL=Casado RENDA_ANUAL=75 ".trim() + System.lineSeparator() +
		"INADIMPLENTE=Sim CASA_PROPRIA=Nao ESTADO_CIVIL=Solteiro RENDA_ANUAL=90 ".trim() + System.lineSeparator() ;

	public static void main( String[ ] args ) throws IOException {
		InputStream model = maxentTrainer( TRAINER_SET, ITERATION, CUTOFF );
		//classifierMaxent( model, CLASSIFIER_SET.split( System.lineSeparator() ) );
		//classificar(model, "INADIMPLENTE=? CASA_PROPRIA=Nao ESTADO_CIVIL=Casado RENDA_ANUAL=120 ".trim());
		//classificar(model, "INADIMPLENTE=? CASA_PROPRIA=Nao ESTADO_CIVIL=Solteiro RENDA_ANUAL=85 ".trim());
		classificar(model, "INADIMPLENTE=? CASA_PROPRIA=Nao ESTADO_CIVIL=Solteiro RENDA_ANUAL=70 ".trim());
	}

	private static void classificar(InputStream is, String documento)throws IOException {
		DataReader dataReader = new PlainTextFileDataReader( is );
		AbstractModel model = new GISModelReader( dataReader ).getModel();
		
		documento = documento.replaceAll( "  +", " " ).trim();
		double[ ] ocs = model.eval( documento.split( " " ) );

		//System.out.print( model.getOutcome( 2 ) + " : " + ocs[ 2 ] + "\t" );
		
		for ( int i = 0; i < ocs.length; i++ ) {
			System.out.print( model.getOutcome( i ) + " : " + ocs[ i ] + "\t" );
		}
		
	}
	
	private static void classifierMaxent( InputStream is, String[ ] documents ) throws IOException {
		DataReader dataReader = new PlainTextFileDataReader( is );
		AbstractModel model = new GISModelReader( dataReader ).getModel();

		int index = 0;
		System.out.println();
		for ( String context : documents ) {
			context = context.replaceAll( "  +", " " ).trim();
			double[ ] ocs = model.eval( context.split( " " ) );

			System.out.print( index++ + "\t|\t" );
			for ( int i = 0; i < ocs.length; i++ ) {
				System.out.print( model.getOutcome( i ) + " : " + ocs[ i ] + "\t" );
			}
			System.out.println();
		}
	}

	private static InputStream maxentTrainer( String corpus, int iteration, int cutoff ) throws IOException {
		ObjectStream< Event > es = new ByteArrayEventStream( new ByteArrayInputStream( corpus.getBytes() ) );
		TwoPassDataIndexer data = new TwoPassDataIndexer( es, cutoff );
		AbstractModel model = GIS.trainModel( iteration, data );

		ByteArrayOutputStream writer = new ByteArrayOutputStream();
		BufferedWriter buffer = new BufferedWriter( new OutputStreamWriter( writer ) );
		new PlainTextGISModelWriter( model, buffer ).persist();
		buffer.close();
		return new ByteArrayInputStream( writer.toByteArray() );
	}
}