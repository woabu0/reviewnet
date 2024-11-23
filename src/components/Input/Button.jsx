function Button(props) {
  return (
    <div className="bg-[#3772FF] text-white text-[18px] rounded-[8px] h-[48px] xl:h-[40px] 2xl:h-[48px] flex items-center justify-center s">
      <button type="submit" className="w-full h-full">
        {props.title}
      </button>
    </div>
  );
}
export default Button;
