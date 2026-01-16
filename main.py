"""
=============================================================================
Tutor IA Socrático - Punto de Entrada Principal
=============================================================================
CLI para el sistema de tutoría con RAG.

Comandos disponibles:
- ingest: Procesa PDFs y los almacena en el vector store
- search: Búsqueda semántica en documentos
- stats: Estadísticas del vector store
=============================================================================
"""

import sys
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from config import paths, chunking, model, chroma, validate_config
from src.ingestion import PDFLoader, DocumentChunker, VectorStoreManager
from src.ingestion.embeddings import create_embedding_generator

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Consola Rich para output formateado
console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Activar modo verbose")
def cli(verbose: bool):
    """
    🎓 Tutor IA Socrático - Sistema de Tutoría Inteligente
    
    Asistente educativo basado en RAG que utiliza el método socrático
    para guiar el aprendizaje.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]Modo verbose activado[/yellow]")


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Buscar PDFs recursivamente")
@click.option("--replace", is_flag=True, help="Reemplazar documentos existentes")
def ingest(source: str, recursive: bool, replace: bool):
    """
    📥 Ingesta documentos PDF al vector store.
    
    SOURCE puede ser un archivo PDF o un directorio con PDFs.
    
    Ejemplos:
    
        python main.py ingest documento.pdf
        
        python main.py ingest data/pdfs/ --recursive
    """
    source_path = Path(source)
    
    console.print(Panel(
        f"[bold blue]Ingesta de Documentos[/bold blue]\n"
        f"Fuente: {source_path}\n"
        f"Recursivo: {'Sí' if recursive else 'No'}",
        title="📥 RAG Pipeline"
    ))
    
    # Validar configuración
    config_status = validate_config()
    if not any([config_status["google_api_key"], config_status["openai_api_key"]]):
        console.print("[red]❌ Error: Configura GOOGLE_API_KEY u OPENAI_API_KEY en .env[/red]")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Inicializar componentes
            task = progress.add_task("Inicializando componentes...", total=None)
            
            loader = PDFLoader()
            chunker = DocumentChunker(
                chunk_size=chunking.chunk_size,
                overlap=chunking.chunk_overlap
            )
            embeddings = create_embedding_generator()
            store = VectorStoreManager(
                persist_directory=paths.CHROMA_DB_DIR,
                embedding_generator=embeddings,
                collection_name=chroma.collection_name
            )
            
            progress.update(task, description="Cargando PDFs...")
            
            # Cargar documentos
            documents = []
            if source_path.is_file():
                documents.append(loader.load(source_path))
            else:
                documents = list(loader.load_directory(source_path, recursive=recursive))
            
            if not documents:
                console.print("[yellow]⚠️ No se encontraron documentos para procesar[/yellow]")
                return
            
            console.print(f"📄 Documentos cargados: {len(documents)}")
            
            # Procesar cada documento
            total_chunks = 0
            for doc in documents:
                progress.update(task, description=f"Procesando {Path(doc.file_path).name}...")
                
                # Eliminar chunks existentes si se solicita reemplazo
                if replace:
                    store.delete_by_source(Path(doc.file_path).name)
                
                # Fragmentar
                chunks = chunker.chunk_document(doc)
                console.print(f"  📦 {Path(doc.file_path).name}: {len(chunks)} chunks")
                
                # Almacenar
                progress.update(task, description=f"Almacenando {Path(doc.file_path).name}...")
                added = store.add_chunks(chunks)
                total_chunks += added
            
            progress.update(task, description="Completado!")
        
        # Mostrar resultado
        console.print(Panel(
            f"[bold green]✅ Ingesta completada[/bold green]\n\n"
            f"Documentos procesados: {len(documents)}\n"
            f"Chunks almacenados: {total_chunks}\n"
            f"Total en vector store: {store.count}",
            title="Resultado"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logger.exception("Error en ingesta")


@cli.command()
@click.argument("query")
@click.option("--n", "-n", default=5, help="Número de resultados")
@click.option("--source", "-s", default=None, help="Filtrar por archivo fuente")
def search(query: str, n: int, source: str):
    """
    🔍 Busca en el vector store.
    
    Realiza búsqueda semántica en los documentos indexados.
    
    Ejemplos:
    
        python main.py search "concepto de integral"
        
        python main.py search "teorema fundamental" -n 3
    """
    console.print(f"\n🔍 Buscando: [bold]{query}[/bold]\n")
    
    try:
        embeddings = create_embedding_generator()
        store = VectorStoreManager(
            persist_directory=paths.CHROMA_DB_DIR,
            embedding_generator=embeddings,
            collection_name=chroma.collection_name
        )
        
        if store.count == 0:
            console.print("[yellow]⚠️ El vector store está vacío. Ejecuta 'ingest' primero.[/yellow]")
            return
        
        # Filtro por fuente si se especifica
        filter_meta = {"source_file": source} if source else None
        
        results = store.search(query, n_results=n, filter_metadata=filter_meta)
        
        if not results:
            console.print("[yellow]No se encontraron resultados[/yellow]")
            return
        
        # Mostrar resultados en tabla
        table = Table(title=f"Resultados ({len(results)})")
        table.add_column("#", style="dim")
        table.add_column("Score", justify="right")
        table.add_column("Fuente")
        table.add_column("Contenido", max_width=60)
        
        for i, result in enumerate(results, 1):
            content_preview = result.content[:100].replace("\n", " ") + "..."
            table.add_row(
                str(i),
                f"{result.score:.3f}",
                result.metadata.get("source_file", "?"),
                content_preview
            )
        
        console.print(table)
        
        # Mostrar contenido completo del primer resultado
        console.print(Panel(
            results[0].content[:500] + "..." if len(results[0].content) > 500 else results[0].content,
            title="📄 Mejor resultado (completo)"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        logger.exception("Error en búsqueda")


@cli.command()
def stats():
    """
    📊 Muestra estadísticas del vector store.
    """
    try:
        embeddings = create_embedding_generator()
        store = VectorStoreManager(
            persist_directory=paths.CHROMA_DB_DIR,
            embedding_generator=embeddings,
            collection_name=chroma.collection_name
        )
        
        stats = store.get_stats()
        
        table = Table(title="📊 Estadísticas del Vector Store")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", justify="right")
        
        table.add_row("Total documentos (chunks)", str(stats.get("total_documents", 0)))
        table.add_row("Total tokens", str(stats.get("total_tokens", 0)))
        table.add_row("Archivos fuente", str(stats.get("unique_sources", 0)))
        table.add_row("Colección", stats.get("collection_name", "?"))
        table.add_row("Directorio", stats.get("persist_directory", "?"))
        
        console.print(table)
        
        # Lista de fuentes
        if stats.get("sources"):
            console.print("\n[bold]Archivos indexados:[/bold]")
            for src in stats["sources"]:
                console.print(f"  📄 {src}")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


@cli.command()
def validate():
    """
    🔧 Valida la configuración del sistema.
    """
    console.print(Panel("[bold]Validación de Configuración[/bold]", title="🔧"))
    
    status = validate_config()
    
    table = Table()
    table.add_column("Componente")
    table.add_column("Estado")
    
    for component, valid in status.items():
        icon = "✅" if valid else "❌"
        table.add_row(component, icon)
    
    console.print(table)
    
    # Mostrar rutas
    console.print(f"\n[bold]Rutas:[/bold]")
    console.print(f"  📁 Proyecto: {paths.ROOT_DIR}")
    console.print(f"  📁 PDFs: {paths.PDF_DIR}")
    console.print(f"  📁 ChromaDB: {paths.CHROMA_DB_DIR}")


if __name__ == "__main__":
    cli()
