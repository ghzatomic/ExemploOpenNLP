package br.com.jonathan;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import opennlp.tools.ml.model.Event;
import opennlp.tools.util.ObjectStream;

public class ByteArrayEventStream implements ObjectStream< Event >{

	protected BufferedReader reader;

	public ByteArrayEventStream( String fileName, String encoding ) throws IOException{
		if ( encoding == null ) {
			reader = new BufferedReader( new FileReader( fileName ) );
		} else {
			reader = new BufferedReader( new InputStreamReader( new FileInputStream( fileName ), encoding ) );
		}
	}

	public ByteArrayEventStream( String fileName ) throws IOException{
		this( fileName, null );
	}

	public ByteArrayEventStream( ByteArrayInputStream file ) throws IOException{
		reader = new BufferedReader( new InputStreamReader( file ) );
	}

	@Override
	public Event read() throws IOException {
		String line;
		if ( ( line = reader.readLine() ) != null ) {
			StringTokenizer st = new StringTokenizer( line );
			String outcome = st.nextToken();
			int count = st.countTokens();
			String[ ] context = new String[ count ];
			for ( int ci = 0; ci < count; ci++ ) {
				context[ ci ] = st.nextToken();
			}

			return new Event( outcome, context );
		} else {
			return null;
		}
	}

	public void close() throws IOException {
		reader.close();
	}

	public static String toLine( Event event ) {
		StringBuilder sb = new StringBuilder();
		sb.append( event.getOutcome() );
		String[ ] context = event.getContext();
		for ( int ci = 0, cl = context.length; ci < cl; ci++ ) {
			sb.append( " " ).append( context[ ci ] );
		}
		sb.append( System.getProperty( "line.separator" ) );
		return sb.toString();
	}

	@Override
	public void reset() throws IOException, UnsupportedOperationException {
		throw new UnsupportedOperationException();
	}
}