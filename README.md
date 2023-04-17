
Forked from :  https://github.com/edornd/mmflood

You can download the MMFlood dataset:
- from Zenodo: [https://zenodo.org/record/6534637](https://zenodo.org/record/6534637)
- from IEEE Dataport: [https://ieee-dataport.org/documents/mmflood-multimodal-dataset-flood-delineation-satellite-imagery](https://ieee-dataport.org/documents/mmflood-multimodal-dataset-flood-delineation-satellite-imagery)

### Structure
The dataset is organized in directories, with a JSON file providing metadata and other information such as the split configuration we selected.
Its internal structure is as follows:

```
activations/
├─ EMSR107-1/
├─ .../
├─ EMSR548-0/
│  ├─ DEM/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
│  ├─ hydro/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
│  ├─ mask/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
│  ├─ s1_raw/
│  │  ├─ EMSR548-0-0.tif
│  │  ├─ EMSR548-0-1.tif
│  │  ├─ ...
activations.json
```
- Each folder is named after the Copernicus EMS code it refers to. Since most of them actually contain more than one area, an incremental counter is added to the name, e.g., `EMSR458-0`, `EMSR458-1` and so on.
- Inside each EMSR folder there are four subfolders containing every available modality and the ground truth, in GeoTIFF format:
    - `DEM`: contains the Digital Elevation Model
    - `hydro`: contains the hydrography map for that region, if present
    - `s1_raw`: contains the Sentinel-1 image in VV-VH format
    - `mask`: contains the flood map, rasterized from EMS polygons
- Every EMSR subregion contains a variable number of tiles. however, for the same area, each modality always contains the same amount of files with the same name. Names have the following format: `<emsr_code>-<emsr_region>_<tile_count>`.
For different reasons (retrieval, storage), areas larger than 2500x2500 pixels were divided in large tiles.
- **Note: Every modality is guaranteed to contain at least one image, except for the hydrography that may be missing.**

Last, the `activations.json` contains informations about each EMS activation, as extracted from the Copernicus Rapid Mapping site, as such:
```json
{
    "EMSR107": {
        ...
    },
    "EMSR548": {
        "title": "Flood in Eastern Sicily, Italy",
        "type": "Flood",
        "country": "Italy",
        "start": "2021-10-27T11:31:00",
        "end": "2021-10-28T12:35:19",
        "lat": 37.435056244442684,
        "lon": 14.954437192250033,
        "subset": "test",
        "delineations": [
            "EMSR548_AOI01_DEL_PRODUCT_r1_VECTORS_v1_vector.zip"
        ]
    },
}
```

### Data specifications
| Image    | Description                                           | Format            | Bands        |
| -------- | ----------------------------------------------------- | ----------------- | ------------ |
| S1 raw   | Georeferenced Sentinel-1 imagery, IW GRD              | GeoTIFF Float32   | 0: VV, 1: VH |
| DEM      | MapZen Digital Elevation Model                        | GeoTIFF Float32   | 0: elevation |
| Hydrogr. | Binary map of permanent water basins, OSM             | GeoTIFF Uint8     | 0: hydro     |
| Mask     | Manually validated ground truth label, Copernicus EMS | GeoTIFF Uint8     | 0: gt        |


### Image metadata
Every image also contains the following contextual information, as GDAL metadata tags:
```xml
<GDALMetadata>
<Item name="acquisition_date">2021-10-31T16:56:28</Item>
  <Item name="code">EMSR548-0</Item>
  <Item name="country">Italy</Item>
  <Item name="event_date">2021-10-27T11:31:00</Item>
</GDALMetadata>
```
- `acquisition_date` refers to the acquisition timestamp of the Sentinel-1 image
- `event_date` refers to official event start date reported by Copernicus EMS


### Goal (myadd)

Use diffusion model to generate new sample which would help us improving the metrics that were in the paper