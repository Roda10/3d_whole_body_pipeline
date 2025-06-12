#!/usr/bin/env python3
"""
Pipeline Results Gallery Collector
Gathers existing rendered results from each pipeline run into organized galleries
"""

import os
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import datetime
import glob
import zipfile

class PipelineResultsCollector:
    """Collects and organizes existing pipeline results into comprehensive galleries"""
    
    def __init__(self, evaluation_results_dir: str = "evaluation_results"):
        self.evaluation_results_dir = Path(evaluation_results_dir)
        self.galleries_dir = self.evaluation_results_dir / "consolidated_galleries"
        self.galleries_dir.mkdir(exist_ok=True)
        
        print(f"🎨 Pipeline Results Collector initialized")
        print(f"Source: {self.evaluation_results_dir}")
        print(f"Output: {self.galleries_dir}")
    
    def find_all_runs(self) -> List[Path]:
        """Find all pipeline run directories"""
        runs = []
        
        # Look for run directories in temp_pipeline subdirectories
        for temp_pipeline_dir in self.evaluation_results_dir.glob("*/temp_pipeline"):
            for frame_dir in temp_pipeline_dir.glob("*"):
                for run_dir in frame_dir.glob("run_*"):
                    if run_dir.is_dir():
                        runs.append(run_dir)
        
        # Also look for direct run directories
        for run_dir in self.evaluation_results_dir.glob("**/run_*"):
            if run_dir.is_dir() and run_dir not in runs:
                runs.append(run_dir)
        
        runs.sort(key=lambda x: x.name)
        print(f"📁 Found {len(runs)} pipeline runs")
        for run in runs:
            print(f"   - {run.relative_to(self.evaluation_results_dir)}")
        
        return runs
    
    def find_input_image(self, run_dir: Path) -> Optional[Path]:
        """Find the input image for this run"""
        input_patterns = [
            "*/temp_input/*.jpg",
            "*/temp_input/*.png", 
            "*/temp_input/*.jpeg",
            "*_img.jpg",
            "*_img.png"
        ]
        
        for pattern in input_patterns:
            input_files = list(run_dir.glob(pattern))
            if input_files:
                return input_files[0]
        
        print(f"   ⚠️  No input image found in {run_dir.name}")
        return None
    
    def find_emoca_results(self, run_dir: Path) -> Dict[str, Optional[Path]]:
        """Find EMOCA detail and coarse results"""
        emoca_results = {'detail': None, 'coarse': None}
        
        emoca_dir = run_dir / "emoca_results"
        if not emoca_dir.exists():
            print(f"   ⚠️  No EMOCA results directory found")
            return emoca_results
        
        # Look for EMOCA output directories (EMOCA_v2_lr_mse_20, etc.)
        for emoca_subdir in emoca_dir.glob("EMOCA*"):
            if emoca_subdir.is_dir():
                # Look for frame directories (like "01_img00", "test200", etc.)
                for frame_dir in emoca_subdir.glob("*"):
                    if frame_dir.is_dir():
                        # Look for specific EMOCA output files
                        detail_file = frame_dir / "out_im_detail.png"
                        coarse_file = frame_dir / "out_im_coarse.png"
                        
                        if detail_file.exists() and emoca_results['detail'] is None:
                            emoca_results['detail'] = detail_file
                        if coarse_file.exists() and emoca_results['coarse'] is None:
                            emoca_results['coarse'] = coarse_file
                        
                        # If standard files not found, look for alternative patterns
                        if emoca_results['detail'] is None or emoca_results['coarse'] is None:
                            detail_files = list(frame_dir.glob("*detail*.jpg")) + list(frame_dir.glob("*detail*.png"))
                            coarse_files = list(frame_dir.glob("*coarse*.jpg")) + list(frame_dir.glob("*coarse*.png"))
                            
                            if detail_files and emoca_results['detail'] is None:
                                emoca_results['detail'] = detail_files[0]
                            if coarse_files and emoca_results['coarse'] is None:
                                emoca_results['coarse'] = coarse_files[0]
                        
                        # Break if we found both
                        if emoca_results['detail'] and emoca_results['coarse']:
                            break
                
                # Break if we found both
                if emoca_results['detail'] and emoca_results['coarse']:
                    break
        
        if emoca_results['detail']:
            print(f"   ✅ EMOCA detail: {emoca_results['detail'].name}")
        else:
            print(f"   ⚠️  EMOCA detail not found")
            
        if emoca_results['coarse']:
            print(f"   ✅ EMOCA coarse: {emoca_results['coarse'].name}")
        else:
            print(f"   ⚠️  EMOCA coarse not found")
        
        return emoca_results
    
    def find_wilor_results(self, run_dir: Path) -> Optional[Path]:
        """Find WiLoR rendered hand results"""
        wilor_dir = run_dir / "wilor_results"
        if not wilor_dir.exists():
            print(f"   ⚠️  No WiLoR results directory found")
            return None
        
        # Look for rendered hand images
        wilor_patterns = [
            "*.jpg",
            "*.png",
            "*_hands*.jpg",
            "*_hands*.png",
            "*_rendered*.jpg", 
            "*_rendered*.png"
        ]
        
        for pattern in wilor_patterns:
            wilor_files = list(wilor_dir.glob(pattern))
            # Filter out input images and parameter files
            for wilor_file in wilor_files:
                if ('input' not in wilor_file.name.lower() and 
                    'parameters' not in wilor_file.name.lower() and
                    wilor_file.suffix.lower() in ['.jpg', '.png', '.jpeg']):
                    print(f"   ✅ WiLoR result: {wilor_file.name}")
                    return wilor_file
        
        print(f"   ⚠️  WiLoR rendered result not found")
        return None
    
    def find_smplestx_results(self, run_dir: Path) -> Optional[Path]:
        """Find SMPLest-X rendered results"""
        smplestx_dir = run_dir / "smplestx_results"
        if not smplestx_dir.exists():
            print(f"   ⚠️  No SMPLest-X results directory found")
            return None
        
        # Look for rendered images in inference output directories
        for inference_dir in smplestx_dir.glob("inference_output_*"):
            if inference_dir.is_dir():
                rendered_files = list(inference_dir.glob("rendered_*.jpg")) + list(inference_dir.glob("rendered_*.png"))
                if rendered_files:
                    print(f"   ✅ SMPLest-X result: {rendered_files[0].name}")
                    return rendered_files[0]
        
        # Look for any rendered files directly in smplestx_results
        rendered_files = list(smplestx_dir.glob("rendered_*.jpg")) + list(smplestx_dir.glob("rendered_*.png"))
        if rendered_files:
            print(f"   ✅ SMPLest-X result: {rendered_files[0].name}")
            return rendered_files[0]
        
        print(f"   ⚠️  SMPLest-X rendered result not found")
        return None
    
    def find_fusion_results(self, run_dir: Path) -> Optional[Path]:
        """Find fusion rendered results (prefer non-mesh_comparison files for stacking)"""
        fusion_dir = run_dir / "fusion_results"
        if not fusion_dir.exists():
            print(f"   ⚠️  No fusion results directory found")
            return None
        
        # Look for render_gallery results first (better for stacking)
        gallery_dir = fusion_dir / "render_gallery"
        if gallery_dir.exists():
            gallery_files = list(gallery_dir.glob("6_fusion_overlay.png")) + list(gallery_dir.glob("7_fusion_standalone.png"))
            if gallery_files:
                print(f"   ✅ Fusion result: {gallery_files[0].name}")
                return gallery_files[0]
        
        # Look for other fusion visualizations (excluding mesh_comparison for now)
        fusion_patterns = [
            "fusion_*.jpg",
            "fusion_*.png",
            "enhanced_*.jpg",
            "enhanced_*.png"
        ]
        
        for pattern in fusion_patterns:
            fusion_files = list(fusion_dir.glob(pattern))
            if fusion_files:
                print(f"   ✅ Fusion result: {fusion_files[0].name}")
                return fusion_files[0]
        
        # Fallback to mesh_comparison if nothing else is found
        mesh_comparison_files = list(fusion_dir.glob("mesh_comparison.png")) + list(fusion_dir.glob("mesh_comparison.jpg"))
        if mesh_comparison_files:
            print(f"   ✅ Fusion result: {mesh_comparison_files[0].name}")
            return mesh_comparison_files[0]
        
        print(f"   ⚠️  Fusion rendered result not found")
        return None
    
    def create_comparison_stack(self, images: Dict[str, Path], output_path: Path) -> bool:
        """Create a horizontal comparison stack of all available images (excluding mesh_comparison)"""
        try:
            valid_images = {}
            
            # Load all available images, but exclude mesh_comparison files
            for name, img_path in images.items():
                if img_path and img_path.exists():
                    # Skip mesh_comparison files from the stack
                    if 'mesh_comparison' in img_path.name.lower():
                        print(f"   📋 Skipping {img_path.name} from comparison stack")
                        continue
                        
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        valid_images[name] = img
            
            if len(valid_images) < 2:
                print(f"   ⚠️  Not enough valid images for comparison stack (need at least 2, got {len(valid_images)})")
                return False
            
            # Resize all images to same height (use the minimum height)
            min_height = min(img.shape[0] for img in valid_images.values())
            resized_images = []
            labels = []
            
            # Define order for consistent layout
            preferred_order = ['input', 'emoca_detail', 'emoca_coarse', 'wilor', 'smplestx', 'fusion']
            
            # Add images in preferred order if available
            for preferred_name in preferred_order:
                if preferred_name in valid_images:
                    img = valid_images[preferred_name]
                    # Resize maintaining aspect ratio
                    h, w = img.shape[:2]
                    new_width = int(w * (min_height / h))
                    resized_img = cv2.resize(img, (new_width, min_height))
                    resized_images.append(resized_img)
                    labels.append(preferred_name.replace('_', ' ').upper())
            
            # Add any remaining images not in preferred order
            for name, img in valid_images.items():
                if name not in preferred_order:
                    h, w = img.shape[:2]
                    new_width = int(w * (min_height / h))
                    resized_img = cv2.resize(img, (new_width, min_height))
                    resized_images.append(resized_img)
                    labels.append(name.replace('_', ' ').upper())
            
            # Stack horizontally
            comparison_stack = np.hstack(resized_images)
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            x_offset = 0
            
            for i, (label, img) in enumerate(zip(labels, resized_images)):
                cv2.putText(comparison_stack, label, (x_offset + 10, 30), 
                           font, font_scale, (255, 255, 255), thickness)
                x_offset += img.shape[1]
            
            # Save comparison stack
            cv2.imwrite(str(output_path), comparison_stack)
            print(f"   📋 Comparison stack includes: {', '.join(labels)}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error creating comparison stack: {e}")
            return False
    
    def collect_run_results(self, run_dir: Path) -> Dict:
        """Collect all results for a single run"""
        run_name = run_dir.name
        print(f"\n📦 Collecting results for: {run_name}")
        
        # Create output directory for this run
        output_dir = self.galleries_dir / run_name
        output_dir.mkdir(exist_ok=True)
        
        # Find all result files
        input_img = self.find_input_image(run_dir)
        emoca_results = self.find_emoca_results(run_dir)
        wilor_result = self.find_wilor_results(run_dir)
        smplestx_result = self.find_smplestx_results(run_dir)
        fusion_result = self.find_fusion_results(run_dir)
        
        # Copy files to output directory
        copied_files = {}
        
        if input_img:
            dest_path = output_dir / f"1_input{input_img.suffix}"
            shutil.copy2(input_img, dest_path)
            copied_files['input'] = dest_path
            print(f"   📋 Copied input: {dest_path.name}")
        
        if emoca_results['detail']:
            dest_path = output_dir / f"2_emoca_detail{emoca_results['detail'].suffix}"
            shutil.copy2(emoca_results['detail'], dest_path)
            copied_files['emoca_detail'] = dest_path
            print(f"   📋 Copied EMOCA detail: {dest_path.name}")
        
        if emoca_results['coarse']:
            dest_path = output_dir / f"3_emoca_coarse{emoca_results['coarse'].suffix}"
            shutil.copy2(emoca_results['coarse'], dest_path)
            copied_files['emoca_coarse'] = dest_path
            print(f"   📋 Copied EMOCA coarse: {dest_path.name}")
        
        if wilor_result:
            dest_path = output_dir / f"4_wilor{wilor_result.suffix}"
            shutil.copy2(wilor_result, dest_path)
            copied_files['wilor'] = dest_path
            print(f"   📋 Copied WiLoR: {dest_path.name}")
        
        if smplestx_result:
            dest_path = output_dir / f"5_smplestx{smplestx_result.suffix}"
            shutil.copy2(smplestx_result, dest_path)
            copied_files['smplestx'] = dest_path
            print(f"   📋 Copied SMPLest-X: {dest_path.name}")
        
        if fusion_result:
            dest_path = output_dir / f"6_fusion{fusion_result.suffix}"
            shutil.copy2(fusion_result, dest_path)
            copied_files['fusion'] = dest_path
            print(f"   📋 Copied Fusion: {dest_path.name}")
        
        # Create comparison stack
        comparison_success = False
        if len(copied_files) >= 2:
            comparison_path = output_dir / "comparison_stack.png"
            comparison_success = self.create_comparison_stack(copied_files, comparison_path)
            if comparison_success:
                print(f"   📋 Created comparison stack: comparison_stack.png")
        
        # Create summary info
        summary = {
            'run_name': run_name,
            'source_directory': str(run_dir.relative_to(self.evaluation_results_dir)),
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'files_found': {
                'input': str(input_img.relative_to(run_dir)) if input_img else None,
                'emoca_detail': str(emoca_results['detail'].relative_to(run_dir)) if emoca_results['detail'] else None,
                'emoca_coarse': str(emoca_results['coarse'].relative_to(run_dir)) if emoca_results['coarse'] else None,
                'wilor': str(wilor_result.relative_to(run_dir)) if wilor_result else None,
                'smplestx': str(smplestx_result.relative_to(run_dir)) if smplestx_result else None,
                'fusion': str(fusion_result.relative_to(run_dir)) if fusion_result else None
            },
            'files_copied': [f.name for f in copied_files.values()],
            'comparison_stack_created': comparison_success,
            'total_files_collected': len(copied_files)
        }
        
        # Save summary
        with open(output_dir / "collection_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Collected {len(copied_files)} files for {run_name}")
        
        return summary
    
    def run_collection(self) -> Dict:
        """Run the complete collection process"""
        print(f"\n{'='*60}")
        print(f"🎨 PIPELINE RESULTS COLLECTION")
        print(f"{'='*60}")
        
        # Find all runs
        runs = self.find_all_runs()
        
        if not runs:
            print(f"❌ No pipeline runs found in {self.evaluation_results_dir}")
            return {'total_runs': 0, 'error': 'No runs found'}
        
        # Collect results for each run
        all_summaries = []
        successful_collections = 0
        
        for run_dir in runs:
            try:
                summary = self.collect_run_results(run_dir)
                all_summaries.append(summary)
                if summary['total_files_collected'] > 0:
                    successful_collections += 1
            except Exception as e:
                print(f"   ❌ Error collecting {run_dir.name}: {e}")
                continue
        
        # Create overall summary
        overall_summary = {
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'total_runs_found': len(runs),
            'successful_collections': successful_collections,
            'output_directory': str(self.galleries_dir),
            'individual_summaries': all_summaries
        }
        
        # Save overall summary
        with open(self.galleries_dir / "overall_collection_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"✅ COLLECTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total runs processed: {len(runs)}")
        print(f"Successful collections: {successful_collections}")
        print(f"Output directory: {self.galleries_dir}")
        print(f"\n📁 Gallery structure:")
        for summary in all_summaries:
            if summary['total_files_collected'] > 0:
                print(f"   - {summary['run_name']}: {summary['total_files_collected']} files")
        
                # Create zip archive
        zip_path = self.create_zip_archive()
        if zip_path:
            overall_summary['zip_archive'] = str(zip_path)

        # Add this line to the final print statements:
        if 'zip_archive' in overall_summary:
            print(f"📦 Zip archive: {overall_summary['zip_archive']}")
        
        return overall_summary

    def create_zip_archive(self) -> Optional[Path]:
        """Create a zip archive of the consolidated galleries"""
        try:
            zip_name = f"consolidated_galleries_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = self.evaluation_results_dir / zip_name
            
            print(f"\n📦 Creating zip archive: {zip_name}")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in self.galleries_dir.rglob('*'):
                    if file_path.is_file():
                        # Add file to zip with relative path
                        arcname = file_path.relative_to(self.evaluation_results_dir)
                        zipf.write(file_path, arcname)
            
            print(f"   ✅ Zip created: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")
            return zip_path
            
        except Exception as e:
            print(f"   ❌ Zip creation failed: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Collect pipeline results into organized galleries')
    parser.add_argument('--evaluation_dir', type=str, default='evaluation_results',
                       help='Path to evaluation results directory')
    
    args = parser.parse_args()
    
    # Run collection
    collector = PipelineResultsCollector(args.evaluation_dir)
    results = collector.run_collection()
    
    if results['successful_collections'] > 0:
        print(f"\n🎯 Access your galleries at: {collector.galleries_dir}")
        print(f"Each run folder contains all collected results plus a comparison stack!")
    else:
        print(f"\n⚠️  No results were successfully collected. Check the directory structure.")

if __name__ == '__main__':
    main()