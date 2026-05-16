"""
Quick script to run the full PolypNet pipeline on a single image and print results to the terminal.
"""
import sys
from pathlib import Path
from pprint import pprint

# Add the parent directory to the path so we can import from webapp
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from webapp.pipeline import _uid, validate_image, detect, classify, segment

def main():
    img_path = BASE_DIR / "PolypsSet" / "train2019" / "Image" / "10.jpg"
    
    if not img_path.exists():
        print(f"Error: Could not find image at {img_path}")
        return

    print("\n" + "="*60)
    print(f"  PolypNet Pipeline Execution")
    print(f"  Image: {img_path.name}")
    print("="*60)
    
    print("\nRunning pipeline (Validation -> Detection -> Segmentation -> Classification)...")
    
    session_id = _uid()
    results = {}
    
    val = validate_image(str(img_path))
    results['validation'] = val
    if not val.get('valid', False):
        print("\nNote: Image rejected by validation.")
    else:
        det = detect(str(img_path), session_id)
        results.update(det)
        
        clf = classify(str(img_path), det.get("detections", []))
        results['classification'] = clf
        
        seg = segment(str(img_path), session_id)
        results.update(seg)
    
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    
    print(f"\n1. Validation Stage:")
    print(f"   - Is Valid Image : {val.get('valid', False)}")
    print(f"   - Confidence     : {val.get('confidence', 0.0):.2f}%")

    print(f"\n2. Detection Stage (YOLO):")
    detections = results.get('detections', [])
    print(f"   - Found          : {len(detections)} polyps")
    for i, d in enumerate(detections):
        conf_val = d.get('confidence', 0.0)
        bbox = d.get('bbox', [None,None,None,None])
        x1,y1,x2,y2 = (bbox + [None,None,None,None])[:4]
        print(f"   - Polyp {i+1} : Label='{d.get('class_name')}', Conf={conf_val*100:.1f}%, Box=[{x1},{y1},{x2},{y2}]")

    print(f"\n3. Segmentation Stage (Att-UNet):")
    seg_keys = {k for k in results if 'seg' in k.lower() or 'mask' in k.lower() or 'overlay' in k.lower()}
    coverage = results.get('coverage_pct', results.get('coverage', 0.0))
    seg_success = results.get('success', bool(seg_keys))
    if seg_success:
        print(f"   - Status         : {'Success' if seg_success else 'Failed'}")
        print(f"   - Mask saved     : {results.get('mask_path')}")
        print(f"   - Overlay saved  : {results.get('overlay_path')}")
        print(f"   - Coverage       : {coverage:.2f}% of image area")
        if (coverage or 0.0) > 0.1 and len(detections) == 0:
            print(f"   ⚠ NOTE: YOLO missed this polyp (image may be too small or low-contrast).")
            print(f"           The U-Net segmentation mask is the reliable result here.")
    else:
        seg_err = results.get('error', '')
        print(f"   - Status         : Failed {'(' + seg_err[:70] + ')' if seg_err else ''}")  # noqa

    print(f"\n4. Classification Stage (Ensemble):")
    if 'classification' in results:
        clf = results['classification']
        label      = clf.get('label', clf.get('final_prediction', 'N/A'))
        conf_score = clf.get('confidence', clf.get('average_confidence', 0.0))
        print(f"   - Combined Result: {label.upper()}")
        print(f"   - Ensemble Conf  : {conf_score*100:.1f}% malignant probability")
        print(f"   - Benign Prob    : {clf.get('benign_prob', 1-conf_score)*100:.1f}%")

        model_scores = clf.get('model_scores', clf.get('model_predictions', {}))
        if model_scores:
            print("\n   - Per-Model Scores (malignant probability):")
            for m_name, score in model_scores.items():
                if isinstance(score, dict):
                    score = score.get('confidence', 0.0)
                print(f"      * {m_name:15}: {float(score)*100:.1f}%")
    else:
        print(f"   - Status         : Skipped.")
        
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
