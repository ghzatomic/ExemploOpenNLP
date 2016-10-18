package br.com.jonathan;

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

public class ClassificadorMaxent{
	public static final Integer ITERATION = 100;
	public static final Integer CUTOFF = 5;
	
	public static final String TRAINER_SET =
		"Play_Tennis=No  Outlook=Sunny    Temperature=Hot  Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=No  Outlook=Sunny    Temperature=Hot  Humidity=Hight  Wind=Strong".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Overcast Temperature=Hot  Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Rain     Temperature=Mild Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Rain     Temperature=Cool Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=No  Outlook=Rain     Temperature=Cool Humidity=Normal Wind=Strong".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Overcast Temperature=Cool Humidity=Normal Wind=Strong".trim() + System.lineSeparator() +
		"Play_Tennis=No  Outlook=Sunny    Temperature=Mild Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Sunny    Temperature=Cool Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Rain     Temperature=Mild Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Sunny    Temperature=Mild Humidity=Normal Wind=Strong".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Overcast Temperature=Mild Humidity=Hight  Wind=Strong".trim() + System.lineSeparator() +
		"Play_Tennis=Yes Outlook=Overcast Temperature=Hot  Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
		"Play_Tennis=No  Outlook=Rain     Temperature=Mild Humidity=Hight  Wind=Strong".trim();
	
	public static final String CLASSIFIER_SET =
			"Play_Tennis=?  Outlook=Sunny    Temperature=Hot  Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Sunny    Temperature=Hot  Humidity=Hight  Wind=Strong".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Overcast Temperature=Hot  Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Rain     Temperature=Mild Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Rain     Temperature=Cool Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Rain     Temperature=Cool Humidity=Normal Wind=Strong".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Overcast Temperature=Cool Humidity=Normal Wind=Strong".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Sunny    Temperature=Mild Humidity=Hight  Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Sunny    Temperature=Cool Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Rain     Temperature=Mild Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Sunny    Temperature=Mild Humidity=Normal Wind=Strong".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Overcast Temperature=Mild Humidity=Hight  Wind=Strong".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Overcast Temperature=Hot  Humidity=Normal Wind=Weak  ".trim() + System.lineSeparator() +
			"Play_Tennis=?  Outlook=Rain     Temperature=Mild Humidity=Hight  Wind=Strong".trim();
	
	public static void main( String[ ] args ) throws IOException {
		InputStream model = maxentTrainer( TRAINER_SET, ITERATION, CUTOFF );
		classifierMaxent( model, CLASSIFIER_SET.split( System.lineSeparator() ) );
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