import os
import Metashape
import time
import logging
from pipeline.utils import setup_logger, move_file, move_all_files, remove_lockfile
from datetime import datetime

class MetashapeProject:
    """
    Handles the creation and management of a Metashape project.
    
    Attributes:
        project_path (str): Path to the Metashape project file.
        doc (Metashape.Document): The Metashape project document object.
    """
    def __init__(self, project_path):
        """
        Initializes a Metashape project and sets the project path.
        
        Args:
            project_path (str): Path to the Metashape project file.
        """
        self.doc = Metashape.Document()
        self.project_path = project_path

    def save(self):
        """
        Saves the current project and removes any lock files, ensuring the project is saved without interruption.
        """
        remove_lockfile(self.project_path)
        self.doc.save(self.project_path)
        time.sleep(5)  # Wait to ensure the save completes properly
        remove_lockfile(self.project_path)

    def add_chunk(self, chunk_name, image_list):
        """
        Adds a chunk to the project and loads the provided images.
        
        Args:
            chunk_name (str): Name of the chunk to be added.
            image_list (list): List of image paths to be loaded into the chunk.

        Returns:
            Metashape.Chunk: The newly created chunk with loaded images.
        """
        chunk = self.doc.addChunk()
        chunk.label = chunk_name
        chunk.addPhotos(image_list, load_xmp_accuracy=True, load_rpc_txt=True)
        logging.info(f"{len(chunk.cameras)} images loaded in chunk: {chunk_name}")
        return chunk

    def import_aeropoints(self, chunk, gcp_csv_path):
        """
        Imports Propeller AeroPoints GCP coordinates from a CSV file into a chunk.
        
        AeroPoints CSV format (from Propeller Cloud export):
            point_name,easting,northing,elevation,accuracy_xy,accuracy_z,crs
        
        The CSV is converted to Metashape marker references and applied to the chunk.
        Photos are matched by clicking the GCP targets visible in the imagery.
        
        Args:
            chunk (Metashape.Chunk): The chunk to add GCP markers to.
            gcp_csv_path (str): Path to the AeroPoints CSV file.
        """
        import csv

        if not os.path.exists(gcp_csv_path):
            logging.warning(f"AeroPoints CSV not found: {gcp_csv_path} — skipping GCP import")
            return

        markers_added = 0
        with open(gcp_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    point_name = row.get('point_name', row.get('name', ''))
                    easting = float(row.get('easting', row.get('x', 0)))
                    northing = float(row.get('northing', row.get('y', 0)))
                    elevation = float(row.get('elevation', row.get('z', 0)))
                    accuracy_xy = float(row.get('accuracy_xy', row.get('horiz_accuracy', 0.02)))
                    accuracy_z = float(row.get('accuracy_z', row.get('vert_accuracy', 0.02)))

                    # Create marker in chunk
                    marker = chunk.addMarker()
                    marker.label = point_name

                    # Set marker location (Metashape uses [x, y, z] in chunk CRS)
                    marker.position = Metashape.Vector([easting, northing, elevation])
                    marker.accuracy = Metashape.Vector([accuracy_xy, accuracy_xy, accuracy_z])
                    markers_added += 1
                except (ValueError, KeyError) as e:
                    logging.warning(f"Skipping AeroPoint row {row}: {e}")

        logging.info(f"Imported {markers_added} AeroPoints GCPs into chunk: {chunk.label}")

        # Optimize alignment using GCPs
        if markers_added > 0:
            chunk.optimizeCameras(fit_f=True, fit_cxcy=True, fit_b1=True, fit_b2=True,
                                  fit_k1k2k3=True, fit_k4=True, fit_p1p2=True, fit_p3=True,
                                  fit_decentering=True, fit_adpglobal=True)
            logging.info(f"Optimized camera alignment using {markers_added} GCPs for chunk: {chunk.label}")

class MetashapeChunkProcessor:
    """
    Processes a Metashape chunk, handling various stages of processing such as aligning photos,
    building point clouds, and generating orthomosaics.
    
    Attributes:
        chunk (Metashape.Chunk): The chunk to be processed.
    """
    def __init__(self, chunk):
        """
        Initializes the processor for a given chunk.
        
        Args:
            chunk (Metashape.Chunk): The chunk to be processed.
        """
        self.chunk = chunk

    def align_photos(self):
        """
        Aligns the photos in the chunk by matching tie points and aligning cameras.
        """
        logging.info(f"Aligning photos for chunk: {self.chunk.label}")
        self.chunk.matchPhotos(downscale=1,reference_preselection=True,keypoint_limit=60000,tiepoint_limit=15000,filter_mask=False,guided_matching=True,mask_tiepoints=False)
        self.chunk.alignCameras()

    def build_depth_maps(self):
        """
        Builds depth maps for the chunk.
        """
        logging.info(f"Building Depth Maps for chunk: {self.chunk.label}")
        self.chunk.buildDepthMaps(downscale=2, filter_mode=Metashape.MildFiltering)

    def build_point_cloud(self):
        """
        Builds a point cloud from the depth maps.
        """
        logging.info(f"Building point cloud for chunk: {self.chunk.label}")
        self.chunk.buildPointCloud()

    def build_model(self):
        """
        Builds a 3D model from the point cloud.
        """
        logging.info(f"Building model for chunk: {self.chunk.label}")
        self.chunk.buildModel(source_data=Metashape.DepthMapsData)

    def smooth_model(self):
        """
        Applies smoothing to the 3D model.
        """
        logging.info(f"Smoothing model for chunk: {self.chunk.label}")
        self.chunk.smoothModel(strength=6)

    def build_orthomosaic(self):
        """
        Builds an orthomosaic from the 3D model.
        """
        logging.info(f"Building orthomosaic for chunk: {self.chunk.label}")
        self.chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ModelData)

    def export_raster(self, export_folder, target_crs="EPSG::4326"):
        """
        Exports the orthomosaic as a raster image to the specified folder.
        
        Args:
            export_folder (str): The folder where the orthomosaic will be saved.
            target_crs (str): Coordinate reference system for the output (default EPSG::4326).
        """
        ortho_path = os.path.join(export_folder, f"{self.chunk.label}_orthomosaic.tif")

        compression = Metashape.ImageCompression()
        compression.tiff_compression = Metashape.ImageCompression.TiffCompressionLZW
        compression.jpeg_quality = 90
        compression.tiff_big = True
        compression.tiff_overviews = True
        compression.tiff_tiled = True

        out_projection = Metashape.OrthoProjection()
        out_projection.type = Metashape.OrthoProjection.Type.Planar
        out_projection.crs = Metashape.CoordinateSystem(target_crs)

        # --- CONDITION FOR THERMAL ---
        is_thermal = "THERMAL" in self.chunk.label.upper()

        self.chunk.exportRaster(path=ortho_path,
                                source_data=Metashape.OrthomosaicData,
                                image_compression=compression,
                                save_alpha=not is_thermal,
                                white_background=True,
                                projection=out_projection)

        logging.info(
        f"Exported orthomosaic for chunk {self.chunk.label} "
        f"(save_alpha={'False' if is_thermal else 'True'}) to {ortho_path}"
    )

class MetashapeProcessor:
    """
    Main processor for handling the execution of Metashape processing tasks.
    
    Attributes:
        input_folder (str): Folder containing the project data.
        gpu_option (str): Specifies which GPU to use ('0', '1', or 'all').
        cpu_enabled (bool): Whether CPU processing is enabled.
        tmp_folder (str): Temporary folder for intermediate files.
        log_file (str): Path to the log file for recording processing information.
    """
    def __init__(self, config, log_file):
        """
        Initializes the processor with the given configuration and log file.
        
        Args:
            config (dict): Configuration dictionary containing input_folder, gpu_option, etc.
            log_file (str): Path to the log file.
        """
        self.input_folder = config["input_folder"]
        self.gpu_option = config["gpu_option"]
        self.cpu_enabled = config["cpu_enabled"]
        self.tmp_folder = config["tmp_folder"]
        self.log_file = log_file
        self.gcp_csv = config.get("gcp_csv", "")  # Optional AeroPoints CSV path
        self.target_crs = config.get("target_crs", "EPSG::4326")  # Output CRS

        # Set up logging
        self.logger = setup_logger(self.log_file)

        # Configure GPUs based on the user's choice
        if self.gpu_option == 'all':
            self.logger.info("Using both GPUs (0 and 1).")
            Metashape.app.gpu_mask = (1 << 0) | (1 << 1)  # Enable GPU 0 and GPU 1
        elif self.gpu_option == '0':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.logger.info("Using GPU 0 only.")
            Metashape.app.gpu_mask = 1 << 0  # Enable only GPU 0
        elif self.gpu_option == '1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            self.logger.info("Using GPU 1 only.")
            Metashape.app.gpu_mask = 1 << 1  # Enable only GPU 1
        else:
            raise ValueError("Invalid GPU option. Use '0', '1', or 'all'.")

        # Optionally enable or disable CPU processing
        Metashape.app.cpu_enable = self.cpu_enabled
        self.logger.info(f"CPU enabled: {self.cpu_enabled}")

    def process_folders(self):
        """
        Processes all folders in the input directory that contain unprocessed data.
        """
        for folder_name in sorted(os.listdir(self.input_folder)):
            folder_path = os.path.join(self.input_folder, folder_name)
            if os.path.isdir(folder_path) and "_unprocessed" in folder_name:
                self.process_unprocessed_folder(folder_path)
        
        # Force log flush
        logging.shutdown()

    def process_unprocessed_folder(self, folder_path):
        """
        Processes an unprocessed folder, including image loading, model generation, and exporting results.
        
        Args:
            folder_path (str): Path to the folder to be processed.
        """
        self.logger.info(f"Processing folder: {folder_path}")
        
        # Create a temporary folder for processing
        base_dir = os.path.basename(os.path.normpath(folder_path))
        tmp_project_folder = os.path.join(self.tmp_folder, base_dir)
        os.makedirs(tmp_project_folder, exist_ok=True)

        # Create an export directory inside the _unprocessed folder
        export_folder = os.path.join(tmp_project_folder, "export")
        os.makedirs(export_folder, exist_ok=True)

        # Define the path for the project file
        project_path = os.path.join(tmp_project_folder, "project.psx")
        project = MetashapeProject(project_path)
        project.save()

        # Locate the "photos" directory within the "_unprocessed" folder
        photos_dir = os.path.join(folder_path, "photos")
        if os.path.isdir(photos_dir):
            for subfolder_name in sorted(os.listdir(photos_dir)):
                subfolder_path = os.path.join(photos_dir, subfolder_name)
                if os.path.isdir(subfolder_path):
                    self.logger.info("Processing RGB and multispectral images by their specific naming convention")
                    image_dict = {
                        "RGB": [os.path.join(subfolder_path, file_name) for file_name in os.listdir(subfolder_path) if file_name.endswith(".JPG")],
                        "THERMAL": [os.path.join(subfolder_path, file_name) for file_name in os.listdir(subfolder_path) if file_name.endswith("_T.tif")],
                        "NIR": [os.path.join(subfolder_path, file_name) for file_name in os.listdir(subfolder_path) if file_name.endswith("MS_NIR.TIF")],
                        "RE": [os.path.join(subfolder_path, file_name) for file_name in os.listdir(subfolder_path) if file_name.endswith("MS_RE.TIF")],
                        "R": [os.path.join(subfolder_path, file_name) for file_name in os.listdir(subfolder_path) if file_name.endswith("MS_R.TIF")],
                        "G": [os.path.join(subfolder_path, file_name) for file_name in os.listdir(subfolder_path) if file_name.endswith("MS_G.TIF")],
                    }

                    for channel, image_list in image_dict.items():
                        try:
                            if not image_list:
                                continue
                            
                            self.logger.info(f"Adding {channel} photos to chunk: {subfolder_name}_{channel}")
                            chunk_name = f"{subfolder_name}_{channel}"
                            chunk = project.add_chunk(chunk_name, image_list)
                            processor = MetashapeChunkProcessor(chunk)

                            processor.align_photos()
                            project.save()

                            # Import AeroPoints GCPs if configured
                            if self.gcp_csv and os.path.exists(self.gcp_csv):
                                project.import_aeropoints(chunk, self.gcp_csv)
                                project.save()

                            processor.build_depth_maps()
                            project.save()

                            processor.build_model()
                            project.save()

                            has_transform = chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation

                            if has_transform:
                                processor.build_point_cloud()
                                project.save()

                                processor.smooth_model()
                                project.save()

                                processor.build_orthomosaic()
                                project.save()

                            processor.export_raster(export_folder, self.target_crs)
                            project.save()
                        except Exception as e:
                            self.logger.error(f"Error processing chunk {subfolder_name}_{channel}: {str(e)}")
                            pass

        try:
            move_all_files(tmp_project_folder, folder_path)
            move_file(self.log_file, folder_path)

            processed_folder = folder_path.replace("_unprocessed", "_processed")
            os.rename(folder_path, processed_folder)
            self.logger.info(f"Renamed folder to: {processed_folder}")
        except Exception as e:
            self.logger.error(f"Error handling files or renaming folder: {str(e)}")
            pass
