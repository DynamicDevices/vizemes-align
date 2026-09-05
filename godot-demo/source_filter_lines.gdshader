shader_type canvas_item;

uniform sampler2D measurements : filter_nearest, repeat_disable;
uniform vec4 line_color : source_color = vec4(0.20, 0.82, 1.0, 1.0);

void fragment() {
	float rows = float(textureSize(measurements, 0).y);
	float row = floor(UV.y * rows);
	float lane_y = fract(UV.y * rows);
	float value = texture(measurements, vec2(UV.x, (row + 0.5) / rows)).r;
	float distance_to_line = abs(lane_y - (1.0 - value));
	float line = 1.0 - smoothstep(0.010, 0.030, distance_to_line);
	float divider = 1.0 - smoothstep(0.0, 0.012, min(lane_y, 1.0 - lane_y));
	vec3 background = mix(vec3(0.035, 0.045, 0.070), vec3(0.10), divider * 0.45);
	COLOR = vec4(mix(background, line_color.rgb, line), 1.0);
}
