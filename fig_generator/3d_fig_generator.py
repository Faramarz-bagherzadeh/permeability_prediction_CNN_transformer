from pathlib import Path
import math
import struct
import zlib

from paraview.simple import *


def embed_png_dpi(path, dpi=300):
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")

    # Insert a pHYs chunk immediately after IHDR; PNG stores density in pixels/meter.
    ihdr_length = struct.unpack(">I", data[8:12])[0]
    ihdr_end = 8 + 12 + ihdr_length
    pixels_per_meter = round(dpi / 0.0254)
    phys_data = struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    phys_chunk = (
        struct.pack(">I", len(phys_data))
        + b"pHYs"
        + phys_data
        + struct.pack(">I", zlib.crc32(b"pHYs" + phys_data) & 0xFFFFFFFF)
    )
    path.write_bytes(data[:ihdr_end] + phys_chunk + data[ihdr_end:])


project_dir = Path(__file__).resolve().parent.parent
input_file = project_dir / "data" / "B51_bag02_hxfb101ms_block_0_1.06_117.0_.vti"
output_dir = project_dir / "fig_generator" / "figs"
output_file = output_dir / "B51_bag02_volume.png"
crop_output_file = output_dir / "B51_bag02_volume_slice_50x50x50.png"

reader = XMLImageDataReader(FileName=[str(input_file)])
reader.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.AxesGrid.Visibility = 0
view.EnableRayTracing = 1
view.BackEnd = "OSPRay raycaster"
view.SamplesPerPixel = 1
view.UseColorPaletteForBackground = 0
view.BackgroundColorMode = "Single Color"
view.Background = [1.0, 1.0, 1.0]
display = Show(reader, view)
display.Representation = "Volume"
ColorBy(display, ("CELLS", "image"))

outline = Outline(Input=reader)
outline_display = Show(outline, view)
outline_display.Representation = "Surface"
outline_display.DiffuseColor = [0.0, 0.0, 0.0]
outline_display.LineWidth = 2.0

array_info = reader.GetDataInformation().GetCellDataInformation().GetArrayInformation("image")
minimum, maximum = array_info.GetComponentRange(0)

lut = GetColorTransferFunction("image")
lut.RescaleTransferFunction(minimum, maximum)
lut.RGBPoints = [
    minimum, 0.78, 0.89, 0.97,
    maximum, 0.43, 0.70, 0.95,
]

opacity = GetOpacityTransferFunction("image")
opacity.RescaleTransferFunction(minimum, maximum)
if maximum > minimum:
    opacity.Points = [
        minimum, 0.0, 0.5, 0.0,
        maximum, 0.35, 0.5, 0.0,
    ]
else:
    opacity.Points = [minimum, 0.35, 0.5, 0.0]

display.ScalarOpacityFunction = opacity
display.SetScalarBarVisibility(view, lut, True)

view.ViewSize = [2400, 2400]
view.ResetCamera()
camera = GetActiveCamera()
camera.Elevation(25)
camera.Azimuth(30)
camera.OrthogonalizeViewUp()
camera_position = camera.GetPosition()
camera_focal_point = camera.GetFocalPoint()
camera_direction = [
    camera_position[index] - camera_focal_point[index] for index in range(3)
]
camera_direction_length = math.sqrt(sum(component**2 for component in camera_direction))
camera_direction = [
    component / camera_direction_length for component in camera_direction
]
camera_view_up = camera.GetViewUp()
view.StillRender()

SaveScreenshot(
    str(output_file),
    view,
    ImageResolution=[2400, 2400],
    TransparentBackground=0,
)
embed_png_dpi(output_file, 300)

Hide(reader, view)
Hide(outline, view)

sample_size = 50
extent = reader.GetClientSideObject().GetOutputDataObject(0).GetExtent()
sample_voi = []
for axis in range(3):
    extent_min = extent[axis * 2]
    extent_max = extent[axis * 2 + 1]
    cell_count = extent_max - extent_min
    if cell_count < sample_size:
        raise ValueError(f"Dataset axis {axis} is smaller than {sample_size} cells")
    start = extent_min
    sample_voi.extend([start, start + sample_size])

sample = ExtractSubset(Input=reader)
sample.VOI = sample_voi
sample.UpdatePipeline()
if sample.GetDataInformation().GetNumberOfCells() != sample_size**3:
    raise RuntimeError(f"Expected a {sample_size}x{sample_size}x{sample_size} crop")

sample_display = Show(sample, view)
sample_display.Representation = "Volume"
ColorBy(sample_display, ("CELLS", "image"))
sample_display.ScalarOpacityFunction = opacity
sample_display.SetScalarBarVisibility(view, lut, True)

sample_outline = Outline(Input=sample)
sample_outline_display = Show(sample_outline, view)
sample_outline_display.Representation = "Surface"
sample_outline_display.DiffuseColor = [0.0, 0.0, 0.0]
sample_outline_display.LineWidth = 2.0

view.ResetCamera()
camera = GetActiveCamera()
crop_focal_point = camera.GetFocalPoint()
crop_position = camera.GetPosition()
crop_distance = math.sqrt(
    sum(
        (crop_position[index] - crop_focal_point[index]) ** 2
        for index in range(3)
    )
)
camera.SetFocalPoint(crop_focal_point)
camera.SetPosition(
    [
        crop_focal_point[index] + camera_direction[index] * crop_distance
        for index in range(3)
    ]
)
camera.SetViewUp(camera_view_up)
camera.OrthogonalizeViewUp()
view.StillRender()

SaveScreenshot(
    str(crop_output_file),
    view,
    ImageResolution=[2400, 2400],
    TransparentBackground=0,
)
embed_png_dpi(crop_output_file, 300)

print(f"Saved volume render: {output_file}")
print(f"Saved first {sample_size}x{sample_size}x{sample_size} slice: {crop_output_file}")