import AssemblyKeys._ // put this at the top of the file

assemblySettings

organization := "fr.iscpif"

name := "schelling"

version := "1.0"

scalaVersion := "2.10.0"

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
 {
  case PathList("org", "bridj", xs @ _*) => MergeStrategy.first
  case PathList( "META-INF", "MANIFEST.MF" ) => MergeStrategy.discard
  case _ => MergeStrategy.deduplicate
 }
}

mainClass in assembly := Some("fr.iscpif.schelling.Schelling")

libraryDependencies ++= Seq (
   "org.scala-lang" % "scala-swing" % "2.10.0",
   "com.nativelibs4java" % "scalacl_2.10" % "0.3-SNAPSHOT"
)

resolvers ++= Seq(
  "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo",
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Sonatype OSS Repository" at "http://oss.sonatype.org/content/repositories/snapshots"
)
